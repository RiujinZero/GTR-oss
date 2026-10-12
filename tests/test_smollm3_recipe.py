from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from gator.smollm3_recipe import (
    DECAY_UPDATES,
    EFFECTIVE_BATCH_SEQUENCES,
    PEAK_LR,
    STAGE2_UPDATE,
    STAGE3_UPDATE,
    STAGES,
    TOTAL_UPDATES,
    DataTroveTokenizedSource,
    HFStreamingTextSource,
    SmolLM3BatchProvider,
    SourceSpec,
    _nanotron_blend_choice,
    accumulation_steps_for_batch,
    learning_rate_for_update,
    total_updates_for_sequence_length,
    stage_for_update,
)


class _Tokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [int(text)]


class _CharacterTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False, **kwargs):
        values = [ord(character) % 90 for character in text]
        if kwargs.get("truncation"):
            values = values[:kwargs["max_length"]]
        return values


class _NativeDataset:
    def __init__(self, cursor):
        self.cursor = cursor

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]

    def __iter__(self):
        return iter(())


class _CursorSource:
    exact_state_supported = True

    def __init__(self, spec, values):
        self.spec, self.values, self.cursor = spec, values, 0

    def take(self, count):
        result = []
        while len(result) < count:
            result.append(self.values[self.cursor % len(self.values)])
            self.cursor += 1
        return result

    def state_dict(self):
        return {"cursor": self.cursor, "exact": True}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]

    def close(self):
        return None


def test_exact_scaled_boundaries_and_content_bearing_fallbacks():
    assert [len(stage) for stage in STAGES] == [41, 46, 56]
    assert [round(sum(source.weight for source in stage), 5) for stage in STAGES] == [
        1.0,
        1.00005,
        1.00565,
    ]
    assert stage_for_update(STAGE2_UPDATE - 2) == 0
    assert stage_for_update(STAGE2_UPDATE - 1) == 1
    by_name = {source.name: source for stage in STAGES for source in stage}
    csharp = by_name["stack-v2-C-Sharp"]
    assert csharp.repo == "bigcode/starcoderdata"
    assert csharp.config is None
    assert dict(csharp.load_kwargs) == {"data_dir": "c-sharp"}
    assert by_name["stack-v2-Cpp"].repo == "bigcode/starcoderdata"
    assert dict(by_name["stack-v2-Cpp"].load_kwargs) == {"data_dir": "cpp"}
    assert by_name["stack-v2-Swift"].repo == "bigcode/the-stack-dedup"
    assert by_name["stack-v2-Swift"].config is None
    assert dict(by_name["stack-v2-Swift"].load_kwargs) == {"data_dir": "data/swift"}
    assert by_name["stack-edu-SQL"].repo == "bigcode/starcoderdata"
    assert dict(by_name["stack-edu-SQL"].load_kwargs) == {"data_dir": "sql"}
    assert by_name["pull-requests"].repo == "bigcode/starcoder2data-extras"
    assert by_name["pull-requests"].config == "issues"
    assert by_name["github-issues"].config == "issues"
    assert by_name["kaggle"].config == "kaggle"
    assert by_name["jupyter-scripts"].config == "kaggle"
    assert by_name["megamath-web-pro"].repo == "HuggingFaceTB/finemath"
    assert by_name["openmathinstruct-2"].text_columns == ()
    assert by_name["stackexchange"].repo == "allenai/dolmino-mix-1124"
    assert by_name["stackexchange"].config == "stackexchange"
    assert by_name["openmathinstruct-2"].repo == "nvidia/OpenMathInstruct-2"
    assert by_name["openmathinstruct-2"].split == "train"
    assert by_name["openmathreasoning"].repo == "nvidia/OpenMathReasoning"
    assert by_name["openmathreasoning"].config == "default"
    assert by_name["openmathreasoning"].split == "cot"
    assert by_name["open-codereasoning"].repo == "nvidia/OpenCodeReasoning"
    assert by_name["open-codereasoning"].config == "default"
    assert by_name["open-codereasoning"].split == "split_0"
    assert by_name["problem-solving"].repo == "ostapeno/tinygsm-mind"
    assert by_name["2students"].repo == "ostapeno/tinygsm-mind"
    assert stage_for_update(STAGE3_UPDATE - 2) == 1
    assert stage_for_update(STAGE3_UPDATE - 1) == 2
    assert learning_rate_for_update(500) == PEAK_LR
    assert learning_rate_for_update(STAGE3_UPDATE - 1) == PEAK_LR
    assert learning_rate_for_update(STAGE3_UPDATE + DECAY_UPDATES - 1) == 0.0
    assert learning_rate_for_update(500, peak_lr=8e-5) == 8e-5
    assert learning_rate_for_update(250, peak_lr=8e-5) == 4e-5
    assert learning_rate_for_update(STAGE3_UPDATE - 1, peak_lr=8e-5) == 8e-5
    assert EFFECTIVE_BATCH_SEQUENCES == 32
    assert accumulation_steps_for_batch(1) == 32
    assert accumulation_steps_for_batch(2) == 16
    assert accumulation_steps_for_batch(4) == 8
    assert accumulation_steps_for_batch(8) == 4
    assert total_updates_for_sequence_length(4096) == TOTAL_UPDATES
    assert total_updates_for_sequence_length(2048) == 396_729
    assert stage_for_update(
        2 * STAGE2_UPDATE - 1, sequence_length=2048
    ) == 1
    assert learning_rate_for_update(
        1000, peak_lr=8e-5, sequence_length=2048
    ) == 8e-5


def test_nanotron_blend_uses_deterministic_largest_deficit_schedule():
    counts = [0, 0]
    choices = []
    for sample_index in range(10):
        selected = _nanotron_blend_choice((0.7, 0.3), counts, sample_index)
        choices.append(selected)
        counts[selected] += 1
    assert counts == [7, 3]
    assert choices == [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]


def test_provider_resume_restores_blend_and_source_cursors():
    specs = (
        (SourceSpec("a", "fake", 0.7), SourceSpec("b", "fake", 0.3)),
    ) * 3
    first = SmolLM3BatchProvider(
        _Tokenizer(), context_tokens=3, output_slots=2, seed=7, sources=specs
    )
    resumed = SmolLM3BatchProvider(
        _Tokenizer(), context_tokens=3, output_slots=2, seed=999, sources=specs
    )
    try:
        first.sources = tuple(
            tuple(
                _CursorSource(spec, [index * 100 + value for value in range(20)])
                for index, spec in enumerate(stage)
            )
            for stage in specs
        )
        first.sample_cpu(4)
        state = first.state_dict()
        expected = first.sample_cpu(8)
        resumed.sources = tuple(
            tuple(
                _CursorSource(spec, [index * 100 + value for value in range(20)])
                for index, spec in enumerate(stage)
            )
            for stage in specs
        )
        resumed.load_state_dict(state)
        actual = resumed.sample_cpu(8)
        assert torch.equal(actual.context_ids, expected.context_ids)
        assert torch.equal(actual.target_ids, expected.target_ids)
        assert resumed.counts == first.counts
    finally:
        first.close()
        resumed.close()


def test_provider_context_growth_preserves_exact_source_cursors():
    specs = (
        (SourceSpec("a", "fake", 0.7), SourceSpec("b", "fake", 0.3)),
    ) * 3
    first = SmolLM3BatchProvider(
        _Tokenizer(), context_tokens=3, output_slots=2, seed=7, sources=specs
    )
    grown = SmolLM3BatchProvider(
        _Tokenizer(), context_tokens=7, output_slots=2, seed=7, sources=specs
    )
    try:
        values = list(range(200))
        first.sources = tuple(
            tuple(_CursorSource(spec, values) for spec in stage)
            for stage in specs
        )
        first.sample_cpu(4)
        state = first.state_dict()
        grown.sources = tuple(
            tuple(_CursorSource(spec, values) for spec in stage)
            for stage in specs
        )

        try:
            grown.load_state_dict(state)
        except ValueError:
            pass
        else:
            raise AssertionError("window growth must require explicit approval")

        grown.load_state_dict(state, allow_window_resize=True)
        assert grown.blend_positions == first.blend_positions
        assert grown.stage_counts == first.stage_counts
        assert grown.counts == first.counts
        for grown_stage, first_stage in zip(
            grown.sources, first.sources, strict=True
        ):
            for grown_source, first_source in zip(
                grown_stage, first_stage, strict=True
            ):
                assert grown_source.cursor == first_source.cursor
    finally:
        first.close()
        grown.close()


def test_stage_transition_continues_reused_logical_source_cursor():
    specs = (
        (SourceSpec("a", "fake", 1.0),),
        (SourceSpec("a", "fake", 0.4), SourceSpec("b", "fake", 0.6)),
        (SourceSpec("a", "fake", 1.0),),
    )
    provider = SmolLM3BatchProvider(
        _Tokenizer(), context_tokens=1, output_slots=1, sources=specs
    )
    try:
        provider.sources = tuple(
            tuple(_CursorSource(spec, list(range(20))) for spec in stage)
            for stage in specs
        )
        provider.sources[0][0].take(5)
        scaled_stage2 = (
            STAGE2_UPDATE * 4096 + provider.sequence_length - 1
        ) // provider.sequence_length - 1
        provider.set_progress(update=scaled_stage2, micro_step=0)
        assert provider.sources[1][0].cursor == 5
        assert provider.sources[1][1].cursor == 0
    finally:
        provider.close()


def test_structured_rows_are_formatted_without_serializing_metadata(tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    source = HFStreamingTextSource(
        SourceSpec("structured", "fake", 1.0, text_columns=()),
        _CharacterTokenizer(),
        eos_token_id=99,
        cache_root=tmp_path,
        executor=executor,
        chunk_tokens=16,
        prefetch_tokens=4,
        max_document_chars=100,
    )
    try:
        text = source._text(
            {
                "question": "Two plus two?",
                "responses": [{"response": "Four."}],
                "repo_info": {"id": "metadata-must-not-be-training-text"},
            }
        )
        assert text == "Two plus two?\n\nFour."
        assert "metadata" not in text
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        source.close()


def test_raw_hf_source_bounds_ram_and_prefetches_next_chunk_to_disk(tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    source = HFStreamingTextSource(
        SourceSpec("tiny", "fake", 1.0),
        _CharacterTokenizer(),
        eos_token_id=99,
        cache_root=tmp_path / "first",
        executor=executor,
        chunk_tokens=8,
        prefetch_tokens=3,
        max_document_chars=100,
    )
    source._iterator = iter(
        [{"text": "abcdefghijk"}, {"text": "lmnopqrstuv"}, {"text": "wxyzabcdefg"}]
    )
    try:
        assert len(source.take(6)) == 6
        assert len(source._current) <= 8
        assert len(source._document_tokens) <= 8
        state = source.state_dict()
        assert len(state["current"]) <= 8
        assert len(state["document_tokens"]) <= 8
        assert len(state["prefetched"]) <= 8
        assert source.truncated_documents > 0
        expected = source.take(8)

        resumed_executor = ThreadPoolExecutor(max_workers=1)
        resumed = HFStreamingTextSource(
            SourceSpec("tiny", "fake", 0.25),
            _CharacterTokenizer(),
            eos_token_id=99,
            cache_root=tmp_path / "resumed",
            executor=resumed_executor,
            chunk_tokens=8,
            prefetch_tokens=3,
            max_document_chars=100,
        )
        try:
            resumed.load_state_dict(state)
            assert resumed._dataset is None
            assert resumed._iterator is None
            assert resumed.take(8) == expected
        finally:
            resumed_executor.shutdown(wait=True, cancel_futures=True)
            resumed.close()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        source.close()


def test_hf_stream_eviction_preserves_native_cursor_and_releases_reader(tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    source = HFStreamingTextSource(
        SourceSpec("tiny", "fake", 1.0),
        _CharacterTokenizer(),
        eos_token_id=99,
        cache_root=tmp_path,
        executor=executor,
        chunk_tokens=8,
        prefetch_tokens=3,
        max_document_chars=100,
    )
    source._dataset = _NativeDataset(cursor=37)
    source._iterator = iter(source._dataset)
    try:
        assert source.release_stream()
        assert source._dataset is None
        assert source._iterator is None
        assert source._native_state == {"cursor": 37}
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        source.close()


def test_provider_lru_caps_live_hf_readers(tmp_path):
    specs = tuple(
        tuple(SourceSpec(f"source-{index}", "fake", 1.0 / 3) for index in range(3))
        for _ in range(3)
    )
    provider = SmolLM3BatchProvider(
        _CharacterTokenizer(),
        context_tokens=1,
        output_slots=1,
        sources=specs,
        stream_cache_dir=tmp_path,
        chunk_tokens=8,
        prefetch_tokens=3,
        max_document_chars=100,
        max_open_streams=2,
        max_host_ram_gb=0,
        memory_log_every_seconds=0,
    )
    first_stage = provider.sources[0]
    try:
        for index, source in enumerate(first_stage):
            source._dataset = _NativeDataset(cursor=index)
            source._iterator = iter(source._dataset)
            provider._after_source_take(source)
        assert sum(source.stream_open for source in first_stage) == 2
        assert not first_stage[0].stream_open
        assert first_stage[0]._native_state == {"cursor": 0}
        assert provider.stream_evictions == 1
    finally:
        provider.close()


def test_raw_hf_tokenizer_is_bounded_before_materializing_ids(tmp_path):
    class RecordingTokenizer(_CharacterTokenizer):
        def encode(self, text, add_special_tokens=False, **kwargs):
            self.kwargs = kwargs
            return super().encode(text, add_special_tokens=add_special_tokens, **kwargs)

    tokenizer = RecordingTokenizer()
    executor = ThreadPoolExecutor(max_workers=1)
    source = HFStreamingTextSource(
        SourceSpec("tiny", "fake", 1.0), tokenizer, eos_token_id=99,
        cache_root=tmp_path, executor=executor, chunk_tokens=8,
        prefetch_tokens=3, max_document_chars=100,
    )
    source._iterator = iter([{"text": "abcdefghijklmnopqrstuvwxyz"}])
    try:
        source._fill_document_tokens()
        assert tokenizer.kwargs == {"truncation": True, "max_length": 7}
        assert len(source._document_tokens) == 8
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        source.close()


def test_datatrove_reader_is_sequential_bounded_and_exactly_resumable(tmp_path):
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "000.metadata").write_text("tokenizer|4\n", encoding="utf-8")
    np.asarray([1, 2, 3], dtype="<u4").tofile(folder / "000.ds")
    source = DataTroveTokenizedSource(SourceSpec("a", "fake", 0.1), tmp_path)
    assert source.take(5) == [1, 2, 3, 1, 2]
    state = source.state_dict()
    resumed = DataTroveTokenizedSource(SourceSpec("a", "fake", 0.9), tmp_path)
    try:
        resumed.load_state_dict(state)
        assert resumed.take(3) == [3, 1, 2]
    finally:
        source.close()
        resumed.close()
