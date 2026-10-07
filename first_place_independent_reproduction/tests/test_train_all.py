from argparse import Namespace

from first_place_repro.train_all import (
    DEFAULT_REMAINING_ORDER,
    build_command,
    selected_members,
    selected_versions,
)


def arguments(**overrides):
    values = dict(
        data_root="/data",
        output_root="/output",
        fold_file="/output/folds.csv",
        encoder_checkpoint="/weights/model.safetensors",
        versions=None,
        stage="gate",
        gate_repeat=0,
        gate_fold=1,
        repeats=(0, 1, 2),
        folds=(1, 2, 3, 4, 5),
        seed=7,
        epochs=80,
        batch_size=16,
        validation_batch_size=4,
        workers=4,
        save_every=5,
        run_tag="remaining_inferred_v1",
        activation_offload=True,
        resume=False,
        include_zero_weight=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_default_plan_excludes_current_and_zero_weight_versions():
    assert selected_versions(arguments()) == DEFAULT_REMAINING_ORDER
    assert "0719_V1" not in DEFAULT_REMAINING_ORDER
    assert "0802_V2" not in DEFAULT_REMAINING_ORDER


def test_gate_and_formal_member_counts():
    assert selected_members(arguments()) == [(0, 1)]
    assert len(selected_members(arguments(stage="formal"))) == 15


def test_command_preserves_source_seed_and_schedule():
    command = build_command(arguments(), "0801_V1", 0, 1)
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--schedule-epochs") + 1] == "300"
    assert command[command.index("--epochs") + 1] == "80"
    assert "--activation-offload" in command
