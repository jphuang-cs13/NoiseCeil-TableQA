"""
Concrete implementations of table representations.
"""

import csv
import io
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _parse_csv_row(row_str: str) -> List[str]:
    """
    Parse a CSV row string, properly handling quoted fields with commas.
    
    Args:
        row_str: CSV row as string
        
    Returns:
        List of cell values
    """
    # Use csv.reader to properly parse CSV with quotes
    csv_reader = csv.reader(io.StringIO(row_str), delimiter=',', quotechar='"', escapechar='\\')
    try:
        row = next(csv_reader)
        return [cell.strip() for cell in row]
    except StopIteration:
        return []


def _fill_unnamed_headers(header_row: List[Any]) -> List[str]:
    """
    Helper function to fill empty/blank header cells with 'Unnamed: #' pattern.
    The index starts from 0 for the first cell and increments for each cell,
    regardless of whether the cell is empty or not.
    
    Args:
        header_row: A list of header cells
    
    Returns:
        List of header cells with empty ones replaced by 'Unnamed: {index}'
    """
    result = []
    
    for index, cell in enumerate(header_row):
        cell_str = str(cell).strip()
        # Check if cell is empty, 'nan', 'None', or only whitespace
        if not cell_str or cell_str.lower() in ['nan', 'none', '']:
            result.append(f"Unnamed: {index}")
        else:
            result.append(cell_str)
    
    return result


def _fill_nan_instances_row(row: List[Any]) -> List[str]:
    """
    Helper to normalize instance row cells, filling empty values with 'nan'.

    Rules:
    - None or whitespace-only -> 'nan'
    - 'None' (case-insensitive) -> 'nan'
    - Keep existing 'nan' as 'nan'
    - Otherwise, strip and stringify the value
    """
    normalized: List[str] = []
    for cell in row:
        if cell is None:
            normalized.append("nan")
            continue
        cell_str = str(cell).strip()
        if not cell_str or cell_str.lower() == "none":
            normalized.append("nan")
        else:
            # Preserve explicit 'nan' values; otherwise keep stripped string
            normalized.append(cell_str)
    return normalized


def _is_all_unnamed_row(row_data: List[str]) -> bool:
    """
    Check if a header row consists entirely of 'Unnamed: #' columns.
    
    Args:
        row_data: List of header cell strings
    
    Returns:
        True if all cells match the 'Unnamed: #' pattern, False otherwise
    """
    if not row_data:
        return False
    
    for cell in row_data:
        cell_str = str(cell).strip()
        # Check if cell matches 'Unnamed: #' pattern
        if not cell_str.startswith('Unnamed: '):
            return False
    
    return True


def _generate_separator_row(header_cells: List[str], separator: str = " | ") -> str:
    """
    Generate a separator row with dashes based on header cell display widths.
    Each cell gets padding from the separator (e.g., " | " adds 2 spaces).
    Uses at least 3 dashes per column for Markdown table compatibility.
    
    Args:
        header_cells: List of header cell strings
        separator: Separator used between columns
        
    Returns:
        Separator row string in Markdown table format
    """
    # Calculate separator padding (spaces added around each cell)
    # For separator " | ", each cell gets 1 space before and 1 space after
    if separator == " | ":
        padding = 2
    else:
        # For other separators, assume no padding
        padding = 0
    
    separator_parts = []
    for cell in header_cells:
        # Display width = cell content + padding
        display_width = len(cell) + padding
        width = max(3, display_width)
        dashes = "-" * width
        separator_parts.append(dashes)
    
    return "|" + "|".join(separator_parts) + "|"


class TableRepresentation(ABC):
    """
    Abstract base class for table representation models.

    Subclasses should implement methods to build representations of tables.
    """

    @abstractmethod
    def build_representation(self, table: Dict[str, Any], **kwargs) -> Any:
        """
        Build a representation for a single table.

        Args:
            table: Table data dict (from DatasetLoader)
            **kwargs: Additional parameters

        Returns:
            The table representation (e.g., vector, embedding, etc.)
        """
        pass

    @abstractmethod
    def build_representations(self, tables: List[Dict[str, Any]], **kwargs) -> List[Any]:
        """
        Build representations for multiple tables.

        Args:
            tables: List of table data dicts
            **kwargs: Additional parameters

        Returns:
            List of table representations
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str = None):
        """Load the representation model."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the representation model to free resources."""
        pass


class TextTableRepresentation(TableRepresentation):
    """
    Represent table as concatenated text with optional truncation.

    Converts the entire table (header + instances) to a single text string.
    """

    def __init__(self,
                 include_file_name: bool = False,
                 include_sheet_name: bool = False,
                 include_metadata: bool = False,
                 max_length: Optional[int] = None,
                 separator: str = " | "):
        """
        Args:
            include_file_name: Whether to include file_name in representation
            include_sheet_name: Whether to include sheet_name in representation
            include_metadata: Whether to include metadata dict as formatted text
            max_length: Maximum length of the text representation (None for no limit)
            separator: Separator between columns
        """
        self.include_file_name = include_file_name
        self.include_sheet_name = include_sheet_name
        self.include_metadata = include_metadata
        self.max_length = max_length
        self.separator = separator
        self.loaded = True  # No actual model to load

    def load_model(self, model_path: str = None):
        self.loaded = True

    def unload_model(self):
        self.loaded = False

    def build_representation(self, table: Dict[str, Any], **kwargs) -> str:
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        parts = []

        # Add metadata if requested
        if self.include_file_name and 'file_name' in table and table.get('file_name'):
            parts.append(f"File name: {table['file_name']}")
        if self.include_sheet_name and 'sheet_name' in table and table.get('sheet_name'):
            parts.append(f"Sheet name: {table['sheet_name']}")

        # Build table content
        table_lines = []
        header_lines = []  # Separate list for header rows
        
        # Add header (support single or multi-row headers)
        header = table.get('header', [])
        last_header_row = None
        if header:
            if isinstance(header, list) and header:
                if all(isinstance(row, list) for row in header):
                    # Header provided as list of rows
                    for hr in header:
                        filled_header = _fill_unnamed_headers(hr)
                        row_str = self.separator.join(filled_header)
                        header_lines.append("| " + row_str + " |")
                        last_header_row = filled_header
                elif all(isinstance(row, str) for row in header):
                    # Header provided as list of strings; decide if multi-row (CSV-like) or single-row
                    if any(',' in s for s in header):
                        # Treat each string as a header row CSV; split by comma
                        for hs in header:
                            cells = [c.strip() for c in hs.split(',')]
                            filled_header = _fill_unnamed_headers(cells)
                            row_str = self.separator.join(filled_header)
                            header_lines.append("| " + row_str + " |")
                            last_header_row = filled_header
                    else:
                        # Treat as single header row of columns
                        filled_header = _fill_unnamed_headers(header)
                        row_str = self.separator.join(filled_header)
                        header_lines.append("| " + row_str + " |")
                        last_header_row = filled_header
                else:
                    # Fallback: join as single row
                    filled_header = _fill_unnamed_headers(header)
                    row_str = self.separator.join(filled_header)
                    header_lines.append("| " + row_str + " |")
                    last_header_row = filled_header

        # Determine if multi-row header
        is_multi_row_header = len(header_lines) > 1
        
        # Join header rows with space if multi-row, otherwise use newline
        if header_lines:
            if is_multi_row_header:
                # Multi-row header: join with space
                table_lines.append(" ".join(header_lines))
            else:
                # Single-row header: add directly
                table_lines.extend(header_lines)

        # Add separator row after header
        if last_header_row:
            separator_row = _generate_separator_row(last_header_row, self.separator)
            table_lines.append(separator_row)

        # Add instances
        instances = table.get('instances', [])
        for row in instances:
            filled_row = _fill_nan_instances_row(row)
            row_str = self.separator.join(filled_row)
            table_lines.append("| " + row_str + " |")
        
        # Add Table: prefix - use space separator for multi-row header tables
        if table_lines:
            if is_multi_row_header:
                # Multi-row header: join all rows with space
                parts.append(f"Table:\n{' '.join(table_lines)}")
            else:
                # Single-row header: join with newline
                parts.append(f"Table:\n{chr(10).join(table_lines)}")

        # Join all parts
        text = "\n".join(parts)

        # Truncate if needed (max_length=-1 means no truncation, use entire table)
        if self.max_length is not None and self.max_length > 0 and len(text) > self.max_length:
            text = text[:self.max_length - 3] + "..."

        return text

    def build_representations(self, tables: List[Dict[str, Any]], **kwargs) -> List[str]:
        return [self.build_representation(table, **kwargs) for table in tables]

