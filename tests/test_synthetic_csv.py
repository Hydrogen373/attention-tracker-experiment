import csv

import pytest

from generate_synthetic_csv import FIELDNAMES, generate_rows, write_csv
from tracker.datasets import SyntheticHeadDataset


class FakeModel:
    def generate(self, case, max_new_tokens):
        assert max_new_tokens == 128
        return f"answer, for {case.sample_id}\nsecond line"


def test_synthetic_generation_csv_has_all_pairs_and_preserves_responses(tmp_path):
    pairs = SyntheticHeadDataset().load_pairs(30)
    rows = generate_rows(FakeModel(), pairs, max_new_tokens=128)
    output = tmp_path / "synthetic_generation.csv"
    write_csv(rows, output)

    with output.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(FIELDNAMES)
        saved = list(reader)
    assert len(saved) == 60
    for index, (clean, attack) in enumerate(pairs):
        clean_row, attack_row = saved[2 * index:2 * index + 2]
        for row, case in ((clean_row, clean), (attack_row, attack)):
            assert row["pair_id"] == f"head-{index}"
            assert row["sample_id"] == case.sample_id
            assert row["label"] == str(case.label)
            assert row["system_prompt"] == case.system
            assert row["user_prompt"] == case.user
            assert row["generated_response"] == f"answer, for {case.sample_id}\nsecond line"
            assert row["manual_success"] == row["manual_notes"] == ""
        assert clean_row["target_word"] == ""
        assert attack_row["target_word"] == SyntheticHeadDataset.WORDS[index]


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "synthetic_generation.csv"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_csv([], output)

    assert output.read_text(encoding="utf-8") == "existing"


def test_failed_generation_does_not_create_csv(tmp_path):
    class FailingModel:
        calls = 0

        def generate(self, case, max_new_tokens):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("generation failed")
            return "partial answer"

    output = tmp_path / "synthetic_generation.csv"
    with pytest.raises(RuntimeError, match="generation failed"):
        rows = generate_rows(FailingModel(), SyntheticHeadDataset().load_pairs(30), 128)
        write_csv(rows, output)

    assert not output.exists()
