from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from docling.document_converter import DocumentConverter
from transformers import (AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder,
                          RagConfig, RagRetriever)

from DocReader import DocReader
from VectorDataset import VectorDataset


class Client:
    def __init__(
        self,
        ctx_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        q_model_name: str = "facebook/dpr-question_encoder-single-nq-base",
        ds_name: str = "test",
    ) -> None:

        self.ctx_tok = AutoTokenizer.from_pretrained(ctx_model_name)
        self.ctx_enc = DPRContextEncoder.from_pretrained(ctx_model_name)
        self.ctx_tok.add_special_tokens({"pad_token": self.ctx_tok.pad_token})
        self.ctx_window = 512

        self.q_tok = AutoTokenizer.from_pretrained(q_model_name)
        self.q_enc = DPRQuestionEncoder.from_pretrained(q_model_name)

        self.ds = VectorDataset(ds_name)
        self.md_formatter = DocReader()
        pass

    def add_docs(self, docs: List[str]):
        md_outs = (self.md_formatter.read(doc) for doc in docs)
        all_content = []

        for loc, doc in zip(docs, md_outs):
            sections = self.md_formatter.split_md(doc)

            hdr_list = []
            new_tok = []
            for j in sections:
                # sliding window
                new_tok += self._sliding_window_tok(
                    j["content"], window_size=self.ctx_window, overlap_size=16
                )
                hdr_list += [j["header"]] * len(new_tok)

            new_tok = torch.stack(new_tok)

            with torch.no_grad():
                emb = self.ctx_enc(new_tok).pooler_output
            plaintext = self.ctx_tok.batch_decode(new_tok, ignore_special_tokens=True)

            content = [
                {
                    "location": loc,
                    "text": text,
                    "embedding": embedding.numpy(),
                    "label": label,
                    "metadata": {},
                }
                for text, embedding, label in zip(plaintext, emb, hdr_list)
            ]
            all_content.append(content)
        self.ds.add_docs(all_content)

    def _sliding_window_tok(
        self, text: str, window_size: int = 512, overlap_size: int = 32
    ) -> List[torch.Tensor]:
        # Tokenize the text
        tokens = self.ctx_tok(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].long()

        n_later_tok = tokens.shape[1] - window_size
        later_window_size = window_size - overlap_size
        padding_len = -1
        if n_later_tok < 0:
            padding_len = -n_later_tok
        elif n_later_tok % later_window_size != 0:
            # excludes first full-sized window
            leftover = n_later_tok % later_window_size
            padding_len = later_window_size - leftover

        if padding_len != -1:
            pad_tensor = torch.Tensor([[self.ctx_tok.pad_token_id] * padding_len]).long()
            tokens = torch.cat((tokens, pad_tensor), dim=1)

        start = 0
        end = window_size
        tokenized_text = []

        while start < tokens.shape[1]:
            tokenized_text.append(tokens[0, start:end])
            if end == window_size:
                break
            start += (window_size - overlap_size) if start == 0 else window_size
            end = start + window_size
        return tokenized_text

    def retrieve(self, query: str, n_docs: int = 5) -> List[Dict[str, Any]]:
        tok_q = self.q_tok(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        with torch.no_grad():
            np_q = self.q_enc(**tok_q).pooler_output.numpy()

        out = self.ds.faiss_search(np_q, k=n_docs)
        return out
