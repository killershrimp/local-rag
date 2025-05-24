import re
import time
from io import BytesIO
from typing import Dict, List

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


class DocReader:
    def __init__(self) -> None:
        self.converter = DocumentConverter()
        pass

    # TODO: use convert_all
    def read_link(self, link_or_fp: str) -> str:
        """Read a document from a link"""
        return self.converter.convert(link_or_fp).document.export_to_markdown()

    def read_bytes(self, name: str, stream: BytesIO) -> str:
        docling_stream = DocumentStream(name=name, stream=stream)
        return self.converter.convert(docling_stream).document.export_to_markdown()

    def read_batch(self, files_or_streams: List[str | List[str | bytes]]) -> List[str]:
        formatted = [
            i if isinstance(i, str) else DocumentStream(name=i[0], stream=BytesIO(i[1]))
            for i in files_or_streams
        ]
        docs = self.converter.convert_all(formatted)
        return [i.document.export_to_markdown() for i in docs]

    def split_md(self, markdown_text: str):
        # Define a regex pattern for Markdown headers
        header_pattern = r"^(#{1,6})\s+(.*)$"

        # Using finditer to find all headers in the markdown text
        headers = [
            (match.group(0), match.group(2), match.start())
            for match in re.finditer(header_pattern, markdown_text, re.MULTILINE)
        ]
        if not headers:
            headers = [["main", "main", 0]]

        # Initialize a list to hold the sections
        sections = []

        # Iterate through headers and extract sections
        for i in range(len(headers)):
            header_full, header_text, start_index = headers[i]
            start_pos = start_index

            if i < len(headers) - 1:
                # Get the start position of the next header
                next_header_pos = headers[i + 1][2]
                section_content = markdown_text[start_pos:next_header_pos].strip()
            else:
                # Last section goes to the end of the markdown text
                section_content = markdown_text[start_pos:].strip()

            sections.append({"header": header_text, "content": section_content})

        return sections


if __name__ == "__main__":
    doc_reader = DocReader()
    start = time.time()
    markdown_text = doc_reader.read(
        "/Users/ayun/Downloads/_NeurIPS_2025_D_B__Reuse_Bench.pdf"
    )
    end = time.time()
    print(f"total time: {end - start}")
    sections = doc_reader.split_md(markdown_text)
    for section in sections:
        print(f"Header: {section['header']}")
        # print(
        #     f"Content: {section['content'][:100]}..."
        # )  # Print first 100 chars of content
        # print("-" * 40)
