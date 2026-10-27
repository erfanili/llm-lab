"""Chat template shared by SFT, reward modelling, RLHF, and inspection.

    <|endoftext|>System: {system}\n\nUser: {u1}\n\nAssistant: {a1}<|endoftext|>User: {u2}\n\nAssistant: {a2}<|endoftext|>

Plain-text role markers, no new vocabulary: the base model can already be prompted in this
format, so base vs post-trained models are compared on identical inputs. <|endoftext|> ends
each assistant turn (the base model already knows it as "document over"), and is the stop token.

Segments are tokenized separately and concatenated, so the prompt tokens at inference are
exactly the prefix tokens seen in training.
"""

from tokenizers import Tokenizer

EOT = "<|endoftext|>"


class ChatFormat:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.eot = tok.token_to_id(EOT)

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def render(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """-> (token ids, loss mask) with mask = 1 on assistant content and its closing <|endoftext|>."""
        ids, mask = [self.eot], [0]
        system = ""
        for m in messages:
            if m["role"] == "system":
                system = f"System: {m['content'].strip()}\n\n"
            elif m["role"] == "user":
                t = self._enc(f"{system}User: {m['content'].strip()}\n\nAssistant:")
                system = ""
                ids += t
                mask += [0] * len(t)
            elif m["role"] == "assistant":
                t = self._enc(" " + m["content"].strip()) + [self.eot]
                ids += t
                mask += [1] * len(t)
            else:
                raise ValueError(m["role"])
        return ids, mask

    def prompt(self, messages: list[dict]) -> list[int]:
        """Tokens for a conversation ending in a user turn, ready for the assistant to continue."""
        assert messages[-1]["role"] == "user"
        return self.render(messages)[0]

    def response_ids(self, text: str) -> list[int]:
        return self._enc(" " + text.strip()) + [self.eot]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i != self.eot]).strip()
