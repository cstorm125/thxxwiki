# -*- coding: utf-8 -*-
"""
Preprocessing for cleaning wiki
"""
from typing import Collection, Callable
import re
import html


# str->str rules
def fix_html(text: str) -> str:
    """
        List of replacements from html strings in `test`. (code from `fastai`)
        :param str text: text to replace html string
        :return: text where html strings are replaced
        :rtype: str
        :Example:
            >>> fix_html("Anbsp;amp;nbsp;B @.@ ")
            A & B.
    """
    re1 = re.compile(r"  +")
    text = (
        text.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(text))


def rm_brackets(text: str) -> str:
    """
        Remove all empty brackets and artifacts within brackets from `text`.
        :param str text: text to remove useless brackets
        :return: text where all useless brackets are removed
        :rtype: str
        :Example:
            >>> rm_brackets("hey() whats[;] up{*&} man(hey)")
            hey whats up man(hey)
    """
    # remove empty brackets
    new_line = re.sub(r"\(\)", "", text)
    new_line = re.sub(r"\{\}", "", new_line)
    new_line = re.sub(r"\[\]", "", new_line)
    # brakets with only punctuations
    new_line = re.sub(r"\([^a-zA-Z0-9ก-๙]+\)", "", new_line)
    new_line = re.sub(r"\{[^a-zA-Z0-9ก-๙]+\}", "", new_line)
    new_line = re.sub(r"\[[^a-zA-Z0-9ก-๙]+\]", "", new_line)
    # artifiacts after (
    new_line = re.sub(
        r"(?<=\()[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line
    )
    new_line = re.sub(
        r"(?<=\{)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line
    )
    new_line = re.sub(
        r"(?<=\[)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line
    )
    # artifacts before )
    new_line = re.sub(
        r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\))", "", new_line
    )
    new_line = re.sub(
        r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\})", "", new_line
    )
    new_line = re.sub(
        r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\])", "", new_line
    )
    return new_line


def rm_useless_newlines(text: str) -> str:
    """
        Remove multiple newlines in `text`. (code from `fastai`)
        :param str text: text to replace useless newlines
        :return: text where all newlines are reduced to one
        :rtype: str
        :Example:
            >>> rm_useless_newlines("oh\n\nno")
            oh\nno
    """
    return re.sub("(\n){2,}", "\n", text)


def rm_useless_spaces(text: str) -> str:
    """
        Remove multiple spaces in `text`. (code from `fastai`)
        :param str text: text to replace useless spaces
        :return: text where all spaces are reduced to one
        :rtype: str
        :Example:
            >>> rm_useless_spaces("oh         no")
            oh no
    """
    return re.sub(" {2,}", " ", text)


# combine them together
def process_clean(
    text: str,
    pre_rules: Collection[Callable] = [
        fix_html,
        rm_brackets,
        rm_useless_newlines,
        rm_useless_spaces,
    ],
):
    text = text.lower()
    for rule in pre_rules:
        text = rule(text)
    return text
