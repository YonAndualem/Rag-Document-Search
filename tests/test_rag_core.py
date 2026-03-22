"""Unit tests for rag_core utilities (no Ollama or ChromaDB required)."""
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_core import split_into_chunks, build_prompt, build_messages, make_metadata


class TestSplitIntoChunks:
    def test_empty_string_returns_empty(self):
        assert split_into_chunks("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_into_chunks("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = split_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_produces_multiple_chunks(self):
        text = "a " * 500  # 1000 chars, no sentence boundaries → char fallback
        chunks = split_into_chunks(text)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        text = "word " * 300
        chunks = split_into_chunks(text)
        assert len(chunks) >= 2
        end_of_first = chunks[0][-50:]
        start_of_second = chunks[1][:50]
        assert any(c in start_of_second for c in end_of_first.split())

    def test_normalizes_whitespace(self):
        text = "hello    world.\n\nFoo\tbar."
        chunks = split_into_chunks(text)
        assert "  " not in chunks[0]
        assert "\n" not in chunks[0]

    def test_no_empty_chunks(self):
        text = "a " * 400
        chunks = split_into_chunks(text)
        assert all(chunk.strip() for chunk in chunks)

    def test_sentence_aware_splits_on_period(self):
        # Two short sentences that together exceed CHUNK_SIZE should split
        sentence_a = "Word " * 90 + "end."        # ~454 chars
        sentence_b = "Other " * 90 + "done."      # ~546 chars
        text = sentence_a + " " + sentence_b
        chunks = split_into_chunks(text)
        assert len(chunks) >= 2
        # First chunk should contain sentence_a
        assert "end." in chunks[0]

    def test_sentence_aware_overlap_seeds_next_chunk(self):
        # Build text with clear sentence boundaries
        sentence = "This is a sentence that is long enough. "
        text = sentence * 20  # repeated to force multiple chunks
        chunks = split_into_chunks(text)
        if len(chunks) >= 2:
            # The overlap content from end of chunk 0 should appear in chunk 1
            from config import CHUNK_OVERLAP
            overlap_seed = chunks[0][-CHUNK_OVERLAP:]
            assert any(word in chunks[1] for word in overlap_seed.split() if len(word) > 3)

    def test_very_long_single_sentence_char_chunked(self):
        # A single sentence longer than CHUNK_SIZE must still be split
        from config import CHUNK_SIZE
        text = "word " * (CHUNK_SIZE // 4)  # no punctuation → one sentence
        chunks = split_into_chunks(text)
        assert all(len(c) <= CHUNK_SIZE for c in chunks)


class TestBuildPrompt:
    def test_contains_context(self):
        prompt = build_prompt("Some context here.", "What is this?")
        assert "Some context here." in prompt

    def test_contains_question(self):
        prompt = build_prompt("context", "What is the meaning?")
        assert "What is the meaning?" in prompt

    def test_context_before_question(self):
        prompt = build_prompt("MYCONTEXT", "MYQUESTION")
        assert prompt.index("MYCONTEXT") < prompt.index("MYQUESTION")

    def test_returns_string(self):
        assert isinstance(build_prompt("ctx", "q"), str)

    def test_empty_context(self):
        prompt = build_prompt("", "What is this?")
        assert "What is this?" in prompt


class TestBuildMessages:
    def test_returns_list(self):
        msgs = build_messages("ctx", "question")
        assert isinstance(msgs, list)

    def test_last_message_is_user(self):
        msgs = build_messages("ctx", "my question")
        assert msgs[-1]["role"] == "user"
        assert "my question" in msgs[-1]["content"]

    def test_context_included_in_system(self):
        msgs = build_messages("MYCONTEXT", "q")
        assert any("MYCONTEXT" in m["content"] for m in msgs)

    def test_history_included(self):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        msgs = build_messages("ctx", "new question", history)
        contents = [m["content"] for m in msgs]
        assert any("previous question" in c for c in contents)
        assert any("previous answer" in c for c in contents)

    def test_no_history_produces_minimal_messages(self):
        msgs = build_messages("ctx", "q", history=None)
        roles = [m["role"] for m in msgs]
        assert "user" in roles

    def test_history_comes_before_current_query(self):
        history = [{"role": "user", "content": "EARLIER"}]
        msgs = build_messages("ctx", "CURRENT", history)
        contents = [m["content"] for m in msgs]
        earlier_idx = next(i for i, c in enumerate(contents) if "EARLIER" in c)
        current_idx = next(i for i, c in enumerate(contents) if "CURRENT" in c)
        assert earlier_idx < current_idx


class TestMakeMetadata:
    def test_contains_source(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 0)
        assert meta["source"] == "file.pdf"

    def test_contains_uploaded_at(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 0)
        assert meta["uploaded_at"] == "2024-01-01T00:00:00"

    def test_contains_chunk_index(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 5)
        assert meta["chunk_index"] == 5

    def test_returns_dict(self):
        assert isinstance(make_metadata("f", "t", 0), dict)


class TestEmbedText:
    def test_calls_ollama_embeddings(self):
        with patch("rag_core.ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            from rag_core import embed_text
            result = embed_text("hello")
            mock_embed.assert_called_once()
            assert result == [0.1, 0.2, 0.3]


class TestGenerateAnswer:
    def test_calls_ollama_chat(self):
        with patch("rag_core.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "The answer."}}
            from rag_core import generate_answer
            result = generate_answer("some prompt")
            mock_chat.assert_called_once()
            assert result == "The answer."


class TestStreamAnswer:
    def test_yields_tokens(self):
        chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
            {"message": {"content": ""}},   # empty token should be skipped
        ]
        with patch("rag_core.ollama.chat", return_value=iter(chunks)):
            from rag_core import stream_answer
            tokens = list(stream_answer([{"role": "user", "content": "hi"}]))
        assert tokens == ["Hello", " world"]

    def test_returns_generator(self):
        import types
        with patch("rag_core.ollama.chat", return_value=iter([])):
            from rag_core import stream_answer
            result = stream_answer([])
            assert isinstance(result, types.GeneratorType)
