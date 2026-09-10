"""Hermetic CPU checks for the B3b optimization and fixed-validation contract."""

import copy
import hashlib
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src import cnn_rc_training as training
from src import downstream_checkpoint as checkpoints
from src import downstream_metrics as metrics
from src import downstream_run as runs
from src import exd_hox_dataset as datasets
from src.cnn_rc import CNNRC, reverse_complement
from tests.cnn_rc_synthetic_support import SyntheticCase
from tests import test_exd_hox_dataset as public_fixtures
from tests.test_cnn_rc_cli import B3bSyntheticCase


class _TrainingCase(SyntheticCase):
    """Share real synthetic memberships and independent reference arithmetic."""

    def setUp(self):
        super().setUp()
        self.environment = training.configure_runtime("cpu")

    def make_model(self):
        seed = self.run["identity"]["seeds"]["derived_seeds"]["model_initialization"]
        model = CNNRC(seed=seed)
        optimizer = checkpoints.make_optimizer(
            model, self.run["identity"]["configuration"]["resolved_candidate"]["optimizer"])
        return model, optimizer

    def batch(self, start=0, stop=7):
        samples = []
        for index in range(start, min(stop, len(self.training))):
            samples.append(self.training[index])
        return datasets.collate_exd_hox(samples)

    def independent_orientation(self, batch, epoch):
        result = batch.x.clone()
        seed = self.run["identity"]["seeds"]["derived_seeds"]["training_orientation"]
        membership = self.run["identity"]["selection"]["training_membership_hash"]
        for index, metadata in enumerate(batch.metadata):
            if runs.training_orientation(seed, membership, epoch, metadata.logical_example_id):
                result[index] = batch.x[index].flip(dims=(0, 1))
        return result

    def reference_step(self, model, optimizer, batch, epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(self.independent_orientation(batch, epoch))
        mse = (prediction - batch.y).square().mean()
        penalty = 5e-6 * model.W.abs().sum() + 1e-5 * model.W.square().sum()
        total = mse + penalty
        values = {
            "example_count": len(batch.metadata), "update_count": 1,
            "squared_error_sum": float(mse.item()) * len(batch.metadata),
            "regularization_sum": float(penalty.item()),
            "total_loss_sum": float(total.item()),
        }
        total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return values


class TrainingPrimitiveTests(_TrainingCase):
    """Exercise the exact accepted CPU optimization step."""

    def test_one_update_matches_independent_adam_reference_exactly(self):
        batch = self.batch()
        actual, optimizer = self.make_model()
        expected = copy.deepcopy(actual)
        reference_optimizer = torch.optim.Adam(
            expected.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False, foreach=False, fused=False,
            maximize=False, capturable=False, differentiable=False)
        reference = self.reference_step(expected, reference_optimizer, batch, 0)
        result = training.training_step(actual, optimizer, batch, self.run, 0)
        self.assertEqual(checkpoints.state_fingerprint(actual.state_dict()),
                         checkpoints.state_fingerprint(expected.state_dict()))
        self.assertEqual(checkpoints.state_fingerprint(optimizer.state_dict()),
                         checkpoints.state_fingerprint(reference_optimizer.state_dict()))
        self.assertEqual(result, reference)
        self.assertTrue(all(parameter.grad is None for parameter in actual.parameters()))

    def test_exactly_one_forward_penalty_backward_and_adam_step(self):
        model, optimizer = self.make_model()
        batch = self.batch()
        original_backward = torch.Tensor.backward
        backward_calls = []

        def backward(value, *arguments, **keywords):
            backward_calls.append(value.detach().clone())
            return original_backward(value, *arguments, **keywords)

        with (
            patch.object(model, "forward", wraps=model.forward) as forward,
            patch.object(model, "convolution_kernel_penalty", wraps=model.convolution_kernel_penalty) as penalty,
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
            patch.object(torch.Tensor, "backward", backward),
        ):
            result = training.training_step(model, optimizer, batch, self.run, 0)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(penalty.call_count, 1)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(backward_calls), 1)
        self.assertEqual(result["total_loss_sum"], float(backward_calls[0].item()))
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0)

    def test_step_enters_training_mode_and_clears_gradients_at_both_boundaries(self):
        model, optimizer = self.make_model()
        model.eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        original = model.forward

        def training_forward(inputs):
            self.assertTrue(model.training)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            return original(inputs)

        with (
            patch.object(model, "forward", side_effect=training_forward),
            patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as clear,
        ):
            training.training_step(model, optimizer, self.batch(), self.run, 0)
        self.assertEqual(clear.call_count, 2)
        self.assertTrue(all(call.kwargs == {"set_to_none": True} for call in clear.call_args_list))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_penalty_uses_independent_convolution_and_is_not_batch_divided(self):
        batch = self.batch()
        model, optimizer = self.make_model()
        before = model.W.detach().clone()
        expected_penalty = float((5e-6 * before.abs().sum() + 1e-5 * before.square().sum()).item())
        result = training.training_step(model, optimizer, batch, self.run, 0)
        self.assertEqual(result["regularization_sum"], expected_penalty)
        self.assertNotEqual(result["regularization_sum"], expected_penalty / len(batch.metadata))
        self.assertNotEqual(result["regularization_sum"], 2 * expected_penalty)

    def test_final_partial_batch_is_one_complete_adam_update(self):
        model, optimizer = self.make_model()
        batch = self.batch(128, len(self.training))
        self.assertGreater(len(batch.metadata), 0)
        self.assertLess(len(batch.metadata), 128)
        result = training.training_step(model, optimizer, batch, self.run, 0)
        self.assertEqual(result["example_count"], len(self.training) - 128)
        self.assertEqual(result["update_count"], 1)
        for state in optimizer.state.values():
            self.assertEqual(state["step"].item(), 1)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_accumulators_retain_example_weighting(self):
        model, optimizer = self.make_model()
        first = training.training_step(model, optimizer, self.batch(0, 128), self.run, 0)
        last = training.training_step(model, optimizer, self.batch(128, len(self.training)), self.run, 0)
        total_count = first["example_count"] + last["example_count"]
        self.assertEqual(total_count, len(self.training))
        total_sse = first["squared_error_sum"] + last["squared_error_sum"]
        weighted = total_sse / total_count
        mean_of_means = (first["squared_error_sum"] / first["example_count"]
                         + last["squared_error_sum"] / last["example_count"]) / 2
        self.assertNotEqual(weighted, mean_of_means)
        self.assertEqual(first["update_count"] + last["update_count"], 2)

    def test_orientation_clones_inputs_preserves_membership_and_metadata(self):
        batch = self.batch(0, len(self.training))
        before_x = batch.x.clone()
        before_y = batch.y.clone()
        before_metadata = copy.deepcopy(batch.metadata)
        result = training.orient_training_batch(batch, self.run, 1)
        self.assertIsInstance(result, datasets.ExdHoxBatch)
        self.assertNotEqual(result.x.data_ptr(), batch.x.data_ptr())
        self.assertTrue(torch.equal(result.x, self.independent_orientation(batch, 1)))
        self.assertTrue(torch.equal(batch.x, before_x))
        self.assertTrue(torch.equal(result.y, before_y))
        self.assertEqual(result.metadata, before_metadata)
        self.assertEqual(len(result.metadata), len(batch.metadata))
        self.assertEqual({item.logical_example_id for item in result.metadata},
                         {item.logical_example_id for item in batch.metadata})

    def test_orientation_uses_accepted_primitive_and_ignores_model_identity(self):
        batch = self.batch()
        seed = self.run["identity"]["seeds"]["derived_seeds"]["training_orientation"]
        membership = self.run["identity"]["selection"]["training_membership_hash"]
        with patch.object(runs, "training_orientation", wraps=runs.training_orientation) as oriented:
            expected = training.orient_training_batch(batch, self.run, 0)
        self.assertEqual(oriented.call_count, len(batch.metadata))
        for call, metadata in zip(oriented.call_args_list, batch.metadata):
            self.assertEqual(call.args, (seed, membership, 0, metadata.logical_example_id))
        changed = copy.deepcopy(self.run)
        changed["run_id"] = "run_" + "f" * 64
        changed["identity"]["model"]["contract_id"] = "synthetic_other_model"
        changed["identity"]["software"] = {"synthetic_different_software": True}
        actual = training.orient_training_batch(batch, changed, 0)
        self.assertTrue(torch.equal(actual.x, expected.x))
        self.assertEqual(actual.metadata, expected.metadata)

    def test_nonfinite_loss_aborts_before_backward_or_step(self):
        model, optimizer = self.make_model()
        with (
            patch.object(model, "convolution_kernel_penalty", return_value=torch.tensor(float("inf"))),
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
            patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward must not run")),
        ):
            with self.assertRaises(ValueError):
                training.training_step(model, optimizer, self.batch(), self.run, 0)
        step.assert_not_called()

    def test_missing_expected_gradient_aborts_before_step(self):
        model, optimizer = self.make_model()

        def partial_forward(inputs):
            return torch.sigmoid(model.e).reshape(1, 1).expand(inputs.shape[0], 1)

        with (
            patch.object(model, "forward", side_effect=partial_forward),
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            with self.assertRaises(ValueError):
                training.training_step(model, optimizer, self.batch(), self.run, 0)
        step.assert_not_called()

    def test_nonfinite_gradient_aborts_before_step(self):
        model, optimizer = self.make_model()
        handle = model.q.register_hook(lambda value: torch.full_like(value, float("nan")))
        self.addCleanup(handle.remove)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(ValueError):
                training.training_step(model, optimizer, self.batch(), self.run, 0)
        step.assert_not_called()

    def test_post_update_parameter_bn_and_optimizer_finiteness_guards(self):
        for corruption in ("parameter", "bn", "optimizer"):
            with self.subTest(corruption=corruption):
                model, optimizer = self.make_model()
                original_step = optimizer.step

                def corrupted_step(*arguments, **keywords):
                    result = original_step(*arguments, **keywords)
                    if corruption == "parameter":
                        model.W.detach()[0, 0, 0] = float("nan")
                    elif corruption == "bn":
                        model.running_var[0] = float("inf")
                    else:
                        optimizer.state[model.W]["exp_avg"][0, 0, 0] = float("nan")
                    return result

                with patch.object(optimizer, "step", side_effect=corrupted_step):
                    with self.assertRaises(ValueError):
                        training.training_step(model, optimizer, self.batch(), self.run, 0)

    def test_periodic_checkpoint_control_at_one_hundred_updates(self):
        # Isolate the execution control boundary; no invented state is loaded,
        # validated, or published as a scientific checkpoint.
        session = object.__new__(training._TrainingRun)
        session.run = self.run
        session.writer = SimpleNamespace(root=self.root)
        session.state = {"position": {"completed_epoch_count": 0,
                                     "phase": "train", "global_update": 98}}
        published_updates = []

        def advance():
            position = session.state["position"]
            position["global_update"] += 1
            if position["global_update"] == 101:
                position["phase"] = "epoch_complete"
                position["completed_epoch_count"] = 2

        def save():
            published_updates.append(session.state["position"]["global_update"])

        with (
            patch.object(session, "step", side_effect=advance),
            patch.object(session, "save_checkpoint", side_effect=save),
            patch.object(session, "publish_epoch"),
            patch.object(session, "finish", return_value={"finished": True}),
            patch.object(training, "read_artifacts", return_value={"record.json": self.event(epoch=1)}),
        ):
            self.assertEqual(session.execute(), {"finished": True})
        self.assertEqual(published_updates, [100])


class FixedValidationTests(_TrainingCase):
    """Use the complete two-row fixed synthetic validation set."""

    def validation(self):
        return self.data.dataset("validation")

    def snapshot(self, model, optimizer):
        return checkpoints.state_fingerprint({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": checkpoints.capture_rng(),
            "gradients": [parameter.grad for parameter in model.parameters()],
        })

    def test_validation_preserves_model_optimizer_bn_rng_and_prior_mode(self):
        for prior_mode in (True, False):
            with self.subTest(prior_mode=prior_mode):
                model, optimizer = self.make_model()
                training.training_step(model, optimizer, self.batch(), self.run, 0)
                model.train(prior_mode)
                before = self.snapshot(model, optimizer)
                observed = []
                original = model.forward

                def forward(inputs):
                    observed.append((model.training, torch.is_grad_enabled(), inputs.detach().clone()))
                    return original(inputs)

                with patch.object(model, "forward", side_effect=forward):
                    result = training.evaluate_validation(model, optimizer, self.validation(), self.run)
                self.assertEqual(self.snapshot(model, optimizer), before)
                self.assertEqual(model.training, prior_mode)
                self.assertEqual(len(observed), 2)
                self.assertTrue(all(not mode and not gradients for mode, gradients, _ in observed))
                self.assertTrue(torch.equal(observed[1][2], reverse_complement(observed[0][2])))
                self.assertEqual(result["rc_diagnostic"]["violation_count"], 0)

    def test_validation_metrics_receive_complete_arrays_in_fixed_b2_order(self):
        model, optimizer = self.make_model()
        validation = self.validation()
        expected_targets = np.array([validation[index].y.item() for index in range(len(validation))], dtype=np.float64)
        expected_groups = [validation[index].metadata.global_rc_group_id for index in range(len(validation))]
        calls = []
        original = metrics.compute_regression_metrics

        def complete_metrics(targets, predictions, rc_group_ids=None):
            calls.append((np.asarray(targets).copy(), np.asarray(predictions).copy(), list(rc_group_ids)))
            return original(targets, predictions, rc_group_ids)

        with patch.object(metrics, "compute_regression_metrics", side_effect=complete_metrics):
            result = training.evaluate_validation(model, optimizer, validation, self.run)
        self.assertEqual(len(calls), 3)
        for target, prediction, groups in calls:
            np.testing.assert_array_equal(target, expected_targets)
            self.assertEqual(prediction.shape, (len(validation),))
            self.assertEqual(groups, expected_groups)
        self.assertEqual(calls[2][1].dtype, np.float64)
        np.testing.assert_array_equal(calls[2][1], (calls[0][1].astype(np.float64) + calls[1][1].astype(np.float64)) / 2)
        for name, call in zip(("forward", "reverse_complement", "mean"), calls):
            self.assertEqual(result["metrics"][name], original(*call))

    def test_multibatch_validation_metrics_use_complete_arrays_and_partial_batch(self):
        stage = self.root / "many-validation-stage"
        stage.mkdir()
        fixture = public_fixtures.SyntheticPublicFiles(stage, overshoot=True)
        for index in range(127):
            bits = "3e800000" if index % 2 == 0 else "3f400000"
            fixture.logical.append(public_fixtures._logical_row(
                public_fixtures._sequence(2000 + index), "validation", bits=bits))
        fixture.logical.sort(key=lambda row: row["logical_example_id"])
        counts = {"training": 256, "validation": 129, "test": 1}
        groups = {"training": 255, "validation": 129, "test": 1}
        fixture.metadata_changes[public_fixtures.AUDIT_MANIFEST_PATH] = {
            "totals": {"total_row_occurrences": 386}}
        fixture.metadata_changes[public_fixtures.SPLIT_MANIFEST_PATH] = {
            "counts": {"source_occurrences": 386, "logical_examples": 386,
                       "global_rc_group_counts": groups, "per_tf_split_counts": {"Ubx": counts}}}
        per_tf = {}
        for split in counts:
            per_tf[split] = {"logical_examples": counts[split], "rc_groups": groups[split]}
        fixture.contract_changes["expected_counts"] = {
            "source_occurrences": 386, "logical_examples": 386,
            "split_logical_examples": counts, "global_rc_groups": groups,
            "ordering_logical_examples": 256, "level_rows": 4, "per_tf": {"Ubx": per_tf}}
        contract = fixture.write()
        manifest = public_fixtures._stage_manifest(contract)
        (stage / datasets.STAGE_MANIFEST_FILENAME).write_bytes(public_fixtures._json_bytes(manifest) + b"\n")
        stage_id = "exd_hox_training_stage_" + manifest["manifest_hash"]
        for name in ("split_identity_hash", "split_manifest_hash", "subset_set_manifest_hash"):
            self.config["data"][name] = contract[name]
        self.config["data"]["staging_config_sha256"] = hashlib.sha256(
            public_fixtures._json_bytes(contract) + b"\n").hexdigest()
        self.config_path.write_bytes(public_fixtures._json_bytes(self.config) + b"\n")
        self.git("add", "--", "config.json")
        self.git("commit", "--quiet", "-m", "Synthetic fixed-validation membership")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        with patch.object(datasets, "_load_contract", return_value=contract):
            self.run = self.resolve(stage_root=stage, expected_stage_id=stage_id)
            data = datasets.open_public_tf_data(stage, transcription_factor="Ubx", expected_stage_id=stage_id)
        validation = data.dataset("validation")
        self.assertEqual(len(validation), 129)
        self.assertEqual(self.run["identity"]["configuration"]["resolved_candidate"]["validation"]["batch_size"], 128)
        model, optimizer = self.make_model()
        first = torch.full((128, 1), 0.2, dtype=torch.float32)
        last = torch.full((1, 1), 0.9, dtype=torch.float32)
        expected_prediction = np.concatenate((first.numpy().ravel(), last.numpy().ravel())).astype(np.float64)
        expected_target = np.array([validation[index].y.item() for index in range(129)], dtype=np.float64)
        expected_groups = [validation[index].metadata.global_rc_group_id for index in range(129)]
        with (
            patch.object(model, "forward", side_effect=[first, first.clone(), last, last.clone()]) as forward,
            patch.object(metrics, "compute_regression_metrics", wraps=metrics.compute_regression_metrics) as computed,
        ):
            result = training.evaluate_validation(model, optimizer, validation, self.run)
        self.assertEqual(forward.call_count, 4)
        self.assertEqual(computed.call_count, 3)
        for call in computed.call_args_list:
            np.testing.assert_array_equal(call.args[0], expected_target)
            np.testing.assert_array_equal(call.args[1], expected_prediction)
            self.assertEqual(call.args[2], expected_groups)
        expected = metrics.compute_regression_metrics(expected_target, expected_prediction, expected_groups)
        self.assertEqual(result["metrics"]["mean"], expected)
        residuals = expected_prediction - expected_target
        incorrect_batch_mean = float((np.mean(residuals[:128] ** 2) + residuals[128] ** 2) / 2)
        self.assertNotEqual(expected["mse"], incorrect_batch_mean)

    def test_validation_rejects_unaveraged_invariance_failure_before_metrics(self):
        model, optimizer = self.make_model()
        model.train()
        values = [torch.tensor([[0.25], [0.75]], dtype=torch.float32),
                  torch.tensor([[0.35], [0.65]], dtype=torch.float32)]
        with (
            patch.object(model, "forward", side_effect=values),
            patch.object(metrics, "compute_regression_metrics", side_effect=AssertionError("metrics after failed invariance")) as computed,
        ):
            with self.assertRaises(ValueError):
                training.evaluate_validation(model, optimizer, self.validation(), self.run)
        computed.assert_not_called()
        self.assertTrue(model.training)

    def test_validation_tolerance_diagnostics_and_float64_mean(self):
        model, optimizer = self.make_model()
        forward = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
        reverse = forward + torch.tensor([[1e-6], [-2e-6]], dtype=torch.float32)
        difference = np.abs(reverse.numpy().astype(np.float64).ravel()
                            - forward.numpy().astype(np.float64).ravel())
        tolerance = 1e-6 + 1e-5 * np.abs(forward.numpy().astype(np.float64).ravel())
        with patch.object(model, "forward", side_effect=[forward, reverse]):
            result = training.evaluate_validation(model, optimizer, self.validation(), self.run)
        diagnostic = result["rc_diagnostic"]
        self.assertEqual(diagnostic["maximum_absolute_difference"], float(difference.max()))
        self.assertEqual(diagnostic["maximum_normalized_difference"], float((difference / tolerance).max()))
        self.assertEqual(diagnostic["violation_count"], 0)
        targets = np.array([self.validation()[index].y.item() for index in range(len(self.validation()))])
        mean = (forward.numpy().astype(np.float64).ravel() + reverse.numpy().astype(np.float64).ravel()) / 2
        groups = [self.validation()[index].metadata.global_rc_group_id for index in range(len(self.validation()))]
        self.assertEqual(result["metrics"]["mean"], metrics.compute_regression_metrics(targets, mean, groups))

    def test_constant_predictions_preserve_undefined_correlation_reasons(self):
        model, optimizer = self.make_model()
        constant = torch.full((len(self.validation()), 1), 0.5, dtype=torch.float32)
        with patch.object(model, "forward", side_effect=[constant, constant.clone()]):
            result = training.evaluate_validation(model, optimizer, self.validation(), self.run)
        for value in result["metrics"].values():
            self.assertTrue(math.isfinite(value["r2"]))
            self.assertIsNone(value["pearson"])
            self.assertIsNone(value["spearman"])
            self.assertEqual(value["undefined_reasons"], {
                "pearson": "constant_predictions", "spearman": "constant_predictions"})

    def test_validation_rejects_rng_model_and_optimizer_mutations(self):
        for corruption in ("rng", "model", "optimizer"):
            with self.subTest(corruption=corruption):
                model, optimizer = self.make_model()
                training.training_step(model, optimizer, self.batch(), self.run, 0)
                original = model.forward

                def mutated_forward(inputs):
                    result = original(inputs)
                    if corruption == "rng":
                        torch.rand(1)
                    elif corruption == "model":
                        model.running_mean[0] += 0.001
                    else:
                        optimizer.state[model.W]["exp_avg"][0, 0, 0] += 0.001
                    return result

                with patch.object(model, "forward", side_effect=mutated_forward):
                    with self.assertRaises(ValueError):
                        training.evaluate_validation(model, optimizer, self.validation(), self.run)
                self.assertTrue(model.training)

    def test_validation_rejects_training_membership(self):
        model, optimizer = self.make_model()
        with self.assertRaises(ValueError):
            training.evaluate_validation(model, optimizer, self.training, self.run)

    def test_validation_result_contains_metrics_diagnostics_without_row_predictions(self):
        model, optimizer = self.make_model()
        result = training.evaluate_validation(model, optimizer, self.validation(), self.run)
        self.assertEqual(set(result), {"metrics", "rc_diagnostic"})
        self.assertEqual(set(result["metrics"]), {"forward", "reverse_complement", "mean"})
        self.assertEqual(set(result["rc_diagnostic"]), {
            "atol", "rtol", "maximum_absolute_difference", "maximum_normalized_difference", "violation_count"})
        self.assertEqual(result["rc_diagnostic"]["atol"], 1e-6)
        self.assertEqual(result["rc_diagnostic"]["rtol"], 1e-5)
        self.assertLessEqual(result["rc_diagnostic"]["maximum_normalized_difference"], 1)
        self.assertEqual(result["metrics"]["mean"]["sample_count"], len(self.validation()))
        self.assertEqual(result["metrics"]["mean"]["unique_rc_group_count"], self.validation().unique_rc_group_count)


class SelectionTests(SyntheticCase):
    """Exact finite float64 selection rules, with no training or test score."""

    def test_selection_prefers_r2_then_rmse_then_update_then_epoch(self):
        first = self.event(epoch=0, r2=0.25)
        second = self.event(epoch=1, r2=0.5)
        result = training.select_best([first, second])
        self.assertEqual(result["epoch"], 1)
        second["metrics"]["mean"]["r2"] = first["metrics"]["mean"]["r2"]
        second["metrics"]["mean"]["rmse"] = 0.125
        self.assertEqual(training.select_best([first, second])["epoch"], 1)
        second["metrics"]["mean"]["rmse"] = first["metrics"]["mean"]["rmse"]
        self.assertEqual(training.select_best([second, first])["epoch"], 0)
        second["global_update"] = first["global_update"]
        self.assertEqual(training.select_best([second, first])["epoch"], 0)

    def test_selection_has_zero_tolerance_and_returns_a_copy(self):
        first = self.event(epoch=0, r2=0.25)
        second = self.event(epoch=1, r2=math.nextafter(0.25, 1.0))
        before = copy.deepcopy([first, second])
        result = training.select_best([first, second])
        self.assertEqual(result["epoch"], 1)
        result["metrics"]["r2"] = -123.0
        self.assertEqual([first, second], before)

    def test_empty_and_undefined_r2_are_ineligible(self):
        self.assertIsNone(training.select_best([]))
        undefined = self.event()
        undefined["metrics"]["mean"]["r2"] = None
        undefined["metrics"]["mean"]["undefined_reasons"]["r2"] = "constant_targets"
        self.assertIsNone(training.select_best([undefined]))
        valid = self.event(epoch=1, r2=-10.0)
        self.assertEqual(training.select_best([undefined, valid])["epoch"], 1)

    def test_nonfinite_selection_values_fail_closed(self):
        for field in ("r2", "rmse"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    event = self.event()
                    event["metrics"]["mean"][field] = value
                    with self.assertRaises(ValueError):
                        training.select_best([event])


class TrajectoryRecoveryTests(B3bSyntheticCase):
    """Compare real runs from temporary committed production-source copies."""

    def compare_interruption(self, boundary):
        arguments = {}
        for name, value in self.training_keywords().items():
            arguments[name] = str(value) if hasattr(value, "__fspath__") else value
        source = "arguments = " + repr(arguments) + "\nboundary = " + repr(boundary) + "\n"
        source += '''
from unittest.mock import patch
from src import cnn_rc_training as implementation
from src import downstream_checkpoint as checkpoints

class SyntheticInterruption(RuntimeError):
    pass

for name in ("config", "stage_root", "output_root", "attempt_root"):
    arguments[name] = Path(arguments[name])
baseline_arguments = dict(arguments)
baseline_arguments["output_root"] = arguments["output_root"].parent / "baseline_output"
baseline_arguments["attempt_root"] = arguments["attempt_root"].parent / "baseline_attempts"
recorded_steps = []
validation_calls = []
original_step = implementation.training_step
original_validation = implementation.evaluate_validation

def record_step(model, optimizer, batch, run, epoch):
    oriented = implementation.orient_training_batch(batch, run, epoch)
    recorded_steps.append({
        "epoch": epoch,
        "ids": [item.logical_example_id for item in batch.metadata],
        "orientation": checkpoints.state_fingerprint(oriented.x),
    })
    return original_step(model, optimizer, batch, run, epoch)

def record_validation(*values, **keywords):
    validation_calls.append("validation")
    return original_validation(*values, **keywords)

with patch.object(implementation, "training_step", side_effect=record_step), patch.object(
    implementation, "evaluate_validation", side_effect=record_validation
):
    baseline_completion = implementation.train_cnn_rc(**baseline_arguments)
baseline_steps = list(recorded_steps)
assert len(validation_calls) == 2, validation_calls
recorded_steps.clear()
validation_calls.clear()

original_batch_boundary = implementation._TrainingRun.step
original_validate_epoch = implementation._TrainingRun.validate_epoch
original_commit = implementation._TrainingRun.commit_validation
original_finish = implementation._TrainingRun.finish
interrupted = []

def batch_boundary(instance, *values, **keywords):
    result = original_batch_boundary(instance, *values, **keywords)
    position = instance.state["position"]
    if position["current_epoch"] == 0 and position["next_batch_index"] == 1:
        instance.save_checkpoint()
        interrupted.append(boundary)
        raise SyntheticInterruption(boundary)
    return result

def before_validation(instance, *values, **keywords):
    instance.save_checkpoint()
    interrupted.append(boundary)
    raise SyntheticInterruption(boundary)

def during_validation(model, *values, **keywords):
    validation_calls.append("interrupted_validation")
    original_forward = model.forward
    def interrupted_forward(inputs):
        original_forward(inputs)
        interrupted.append(boundary)
        raise SyntheticInterruption(boundary)
    with patch.object(model, "forward", side_effect=interrupted_forward):
        return original_validation(model, *values, **keywords)

def after_commit(instance, *values, **keywords):
    result = original_commit(instance, *values, **keywords)
    interrupted.append(boundary)
    raise SyntheticInterruption(boundary)

def before_completion(instance, *values, **keywords):
    interrupted.append(boundary)
    raise SyntheticInterruption(boundary)

owner = implementation._TrainingRun
method = {
    "batch": "step", "before_validation": "validate_epoch",
    "during_validation": "evaluate_validation", "after_commit": "commit_validation",
    "before_completion": "finish",
}[boundary]
replacement = {
    "batch": batch_boundary, "before_validation": before_validation,
    "during_validation": during_validation, "after_commit": after_commit,
    "before_completion": before_completion,
}[boundary]
if boundary == "during_validation":
    owner = implementation
with patch.object(implementation, "training_step", side_effect=record_step), patch.object(
    implementation, "evaluate_validation", side_effect=record_validation
):
    with patch.object(owner, method, replacement):
        try:
            implementation.train_cnn_rc(**arguments)
        except SyntheticInterruption:
            pass
        else:
            raise AssertionError("The requested boundary was never interrupted.")
assert interrupted == [boundary], interrupted

resolved = runs.resolve_run(
    config_path=arguments["config"], stage_root=arguments["stage_root"],
    transcription_factor=arguments["tf"], level_id=arguments["level_id"],
    expected_stage_id=arguments["expected_stage_id"], downstream_seed=arguments["downstream_seed"],
    candidate_id=arguments["candidate_id"], checkout_root=Path.cwd(),
    expected_software_commit=arguments["expected_software_commit"], source_paths=implementation.SOURCE_PATHS)
data = datasets.open_public_tf_data(arguments["stage_root"], transcription_factor=arguments["tf"],
                                   expected_stage_id=arguments["expected_stage_id"])
accepted_training = data.dataset("training", level_id=arguments["level_id"])
environment = checkpoints.capture_environment()

def loaded_checkpoints(keyword_arguments):
    result = []
    for root_name in ("output_root", "attempt_root"):
        root = keyword_arguments[root_name]
        if root.exists():
            for manifest in sorted(root.rglob("manifest.json")):
                path = manifest.parent
                if path.name.startswith("update-"):
                    payload, envelope = checkpoints.load_checkpoint(
                        path, run=resolved, environment=environment, training=accepted_training)
                    result.append((path, payload, envelope))
    assert result, "No published checkpoint exists."
    return result

phase_order = {"train": 0, "validation_pending": 1, "epoch_complete": 2}
def latest(records):
    return max(records, key=lambda record: (
        record[1]["position"]["global_update"], phase_order[record[1]["position"]["phase"]]))

resume_checkpoint = latest(loaded_checkpoints(arguments))[0]
with patch.object(implementation, "training_step", side_effect=record_step), patch.object(
    implementation, "evaluate_validation", side_effect=record_validation
):
    resumed_completion = implementation.train_cnn_rc(**dict(arguments, resume_checkpoint=resume_checkpoint))
assert recorded_steps == baseline_steps, (recorded_steps, baseline_steps)
expected_validation_calls = 3 if boundary == "during_validation" else 2
assert len(validation_calls) == expected_validation_calls, validation_calls
baseline_records = loaded_checkpoints(baseline_arguments)
resumed_records = loaded_checkpoints(arguments)
baseline_payload = latest(baseline_records)[1]
resumed_payload = latest(resumed_records)[1]
assert checkpoints.state_fingerprint(resumed_payload) == checkpoints.state_fingerprint(baseline_payload)
assert resumed_payload["position"]["phase"] == "epoch_complete"
assert resumed_payload["position"]["completed_epoch_count"] == 2
assert resumed_payload["consumed_validation_event_count"] == 2
assert resumed_payload["consumed_selection_count"] == 2
assert resumed_payload["early_stopping_state"] == {"enabled": False}
assert len(resumed_payload["validation_history"]) == 2
assert resumed_payload["selection_state"] == baseline_payload["selection_state"]
assert baseline_completion["run_id"] == resumed_completion["run_id"] == resolved["run_id"]
for field in ("run_id", "selected_checkpoint_id", "terminal_recovery_checkpoint_id", "validation_event_id",
              "global_update", "consumed_validation_event_count", "consumed_selection_count", "status"):
    assert resumed_completion[field] == baseline_completion[field], field
assert resumed_completion["status"] == "succeeded"
completion_records = implementation.read_artifacts(arguments["output_root"] / resolved["run_id"] / "completion", resolved)
attempt_environment = completion_records["attempt-terminal.json"]["environment"]
assert len(attempt_environment["validation_replays"]) == int(boundary == "during_validation")
assert len(attempt_environment["validation_reuses"]) == int(boundary == "after_commit")
for epoch in range(2):
    expected_ids = [accepted_training[index].metadata.logical_example_id for index in range(len(accepted_training))]
    order = runs.epoch_order(expected_ids, resolved["identity"]["seeds"]["derived_seeds"]["training_data_order"], epoch)
    actual_ids = []
    for step in recorded_steps:
        if step["epoch"] == epoch:
            actual_ids.extend(step["ids"])
    assert actual_ids == [expected_ids[index] for index in order]
    assert len(actual_ids) == len(set(actual_ids)) == len(accepted_training)
    update = (epoch + 1) * resolved["identity"]["budget"]["batches_per_epoch"]
    phases = {record[1]["position"]["phase"] for record in baseline_records
              if record[1]["position"]["global_update"] == update}
    assert {"validation_pending", "epoch_complete"} <= phases
print(json.dumps({"boundary": boundary, "exact_match": True, "committed_validations": 2,
                  "validation_calls": len(validation_calls), "updates": len(recorded_steps)}))
'''
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["exact_match"])
        self.assertEqual(report["committed_validations"], 2)
        self.assertEqual(report["updates"], 4)

    def test_exact_resume_after_nonterminal_batch(self):
        self.compare_interruption("batch")

    def test_exact_resume_before_validation(self):
        self.compare_interruption("before_validation")

    def test_exact_resume_during_frozen_validation(self):
        self.compare_interruption("during_validation")

    def test_exact_resume_after_committed_validation_without_extra_budget(self):
        self.compare_interruption("after_commit")

    def test_exact_resume_before_completion(self):
        self.compare_interruption("before_completion")

    def test_failed_invariance_prevents_committed_selection_and_completion(self):
        arguments = {}
        for name, value in self.training_keywords().items():
            arguments[name] = str(value) if hasattr(value, "__fspath__") else value
        source = "arguments = " + repr(arguments) + "\n"
        source += '''
from unittest.mock import patch
from src import cnn_rc_training as implementation
from src import downstream_checkpoint as checkpoints
for name in ("config", "stage_root", "output_root", "attempt_root"):
    arguments[name] = Path(arguments[name])
original = implementation.evaluate_validation
observed = []
def broken_invariance(model, optimizer, validation, run):
    observed.append((run, validation))
    forward = torch.full((len(validation), 1), 0.2, dtype=torch.float32)
    reverse = torch.full((len(validation), 1), 0.4, dtype=torch.float32)
    with patch.object(model, "forward", side_effect=[forward, reverse]):
        return original(model, optimizer, validation, run)
with patch.object(implementation, "evaluate_validation", side_effect=broken_invariance):
    try:
        implementation.train_cnn_rc(**arguments)
    except ValueError as error:
        assert "invariance" in str(error).lower(), str(error)
    else:
        raise AssertionError("Broken invariance was accepted.")
assert len(observed) == 1
run = observed[0][0]
root = arguments["output_root"] / run["run_id"]
assert not (root / "completion").exists()
assert not list(root.glob("validation_event_*"))
assert not list(root.glob("update-*-epoch_complete-*"))
indexes = sorted(root.glob("index-*/index.json"))
assert indexes
index = runs.strict_json(checkpoints.read_regular(indexes[-1]))
checkpoints.validate_index(index)
assert index["selected_best_checkpoint_id"] is None
data = datasets.open_public_tf_data(arguments["stage_root"], transcription_factor=arguments["tf"],
                                   expected_stage_id=arguments["expected_stage_id"])
pending_path = root / index["entries"][-1]["path"]
payload, envelope = checkpoints.load_checkpoint(
    pending_path, run=run, environment=checkpoints.capture_environment(),
    training=data.dataset("training", level_id=arguments["level_id"]))
assert payload["position"]["phase"] == "validation_pending"
assert payload["validation_history"] == []
assert payload["consumed_validation_event_count"] == payload["consumed_selection_count"] == 0
assert payload["selection_state"] is None
print(json.dumps({"failed_closed": True}))
'''
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_post_forward_numerical_failure_aborts_without_partial_checkpoint(self):
        arguments = {}
        for name, value in self.training_keywords().items():
            arguments[name] = str(value) if hasattr(value, "__fspath__") else value
        source = "arguments = " + repr(arguments) + "\n"
        source += '''
from unittest.mock import patch
from src import cnn_rc_training as implementation
for name in ("config", "stage_root", "output_root", "attempt_root"):
    arguments[name] = Path(arguments[name])
original = implementation.models.CNNRC.forward
observed = []
def training_forward(model, inputs):
    before = model.running_mean.clone()
    result = original(model, inputs)
    observed.append((before, model.running_mean.clone()))
    return result
def invalid_penalty(model, **coefficients):
    return torch.tensor(float("inf"), dtype=torch.float32)
with patch.object(implementation.models.CNNRC, "forward", training_forward), patch.object(
    implementation.models.CNNRC, "convolution_kernel_penalty", invalid_penalty
):
    try:
        implementation.train_cnn_rc(**arguments)
    except ValueError as error:
        assert "Nonfinite" in str(error), str(error)
    else:
        raise AssertionError("A nonfinite post-forward loss was accepted.")
assert len(observed) == 1
assert not torch.equal(observed[0][0], observed[0][1]), "The actual training forward must have updated BN."
roots = list(arguments["output_root"].glob("run_*"))
assert len(roots) == 1
root = roots[0]
assert not list(root.glob("update-*"))
assert not list(root.glob("validation_event_*"))
assert not (root / "completion").exists()
receipts = list(root.glob("attempt_*-terminal/record.json"))
assert len(receipts) == 1
receipt = runs.strict_json(receipts[0].read_bytes())
runs.validate_attempt(receipt)
assert receipt["status"] == "failed"
assert receipt["exit_code"] != 0
print(json.dumps({"aborted_after_one_forward": True}))
'''
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
