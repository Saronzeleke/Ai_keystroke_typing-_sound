"""
requirements.py - find updates for packages in requirements.txt on pypi
https://github.com/cvzi/requirements

Copyright (C) 2021  cvzi <cuzi@openmail.cc>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

__all__ = ["check_files", "verbose", "parse",
           "parse_file", "parse_line", "get_versions"]

import sys
import logging
import subprocess
import re
import tokenize
import packaging.version

PYTHON = sys.executable
COMMENT_RE = re.compile(r"(^|\s+)#.*$")
REQ_RE = re.compile(r"^\s*([-\w]+)\s*([!<>=~]{1,3})\s*(.+)\s*$")
CACHE = {}


def parse_lines(filename):
    """Join lines that end with a backslash, yields (line_number, line)"""
    last_line = None
    with tokenize.open(filename) as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip()
            if last_line:
                line = last_line + " " + line
                last_line = None
            if line.endswith("\\"):
                last_line = line[:-1]
                line = None
            if line:
                yield index, line
        if last_line:
            yield index, last_line


def parse_line(line, line_number=0, valid_clauses=(">=", "==", "~=")):
    """Parse a single line and return (pkg_name, clause, Version, version_str, line_number)"""
    line = COMMENT_RE.sub("", line)
    line = line.strip()
    m = REQ_RE.match(line)
    if m:
        pkg_name, clause, version = m.groups()
        if clause in valid_clauses:
            return (pkg_name, clause, packaging.version.parse(version), version, line_number)
    return None


def parse_file(filename):
    """Parse a requirements.txt file.
    Skip comments and skip maximum version specifiers"""
    for line_number, line in parse_lines(filename):
        p = parse_line(line, line_number)
        if p:
            yield p


def parse(file_content):
    """Parse content of requiremnts.txt file.
    Skip comments and skip maximum version specifiers"""
    for line_number, line in enumerate(file_content.split("\n")):
        p = parse_line(line, line_number)
        if p:
            yield p


def get_versions(pkg_name):
    """Find available versions for the package"""
    if pkg_name not in CACHE:
        args = (PYTHON, "-mpip", "index", "versions", pkg_name)
        r = subprocess.run(args=args, capture_output=True,
                           check=True, text=True)
        text = r.stdout.split("Available versions:")[1].split("\n")[0].strip()
        CACHE[pkg_name] = []
        for v in text.split(", "):
            try:
                CACHE[pkg_name].append(packaging.version.parse(v))
            except packaging.version.InvalidVersion as e:
                logging.warning('Invalid version', exc_info=e)
    return CACHE[pkg_name]


def check_files(filenames=None, verbose=False):
    """Search each file for updates. Expects a list of filenames.
    If no files are provided, defaults to ./requirements.txt
    Returns a dictionary with the current version, available
    versions and the clause (>=, ==, ...) for each package name"""

    if not filenames:
        filenames = ["requirements.txt"]
    elif isinstance(filenames, str):
        filenames = [filenames]

    results = {}
    for filename in filenames:
        if verbose:
            print("######### Changes " + filename + " #########")
        has_updates = False
        line_updates = {}
        for pkg_name, clause, version, version_str, line_number in parse_file(filename):
            if verbose:
                print(pkg_name, end="", flush=True)
            available_versions = sorted(get_versions(pkg_name))
            results[pkg_name] = {
                "clause": clause,
                "current_version": version,
                "current_version_str": version_str,
                "available_versions": available_versions
            }
            if available_versions[-1] > version:
                if verbose:
                    print("\r" + pkg_name, clause,
                          available_versions[-1], "\t# currently", version, flush=True)
                has_updates = True
                line_updates[line_number] = (
                    version_str, str(available_versions[-1]))
            elif verbose:
                print("\r" + " " * len(pkg_name), end="\r", flush=True)

        if verbose:
            if not has_updates:
                print("#👍 No updates found")
            print("")

        if has_updates and verbose:
            # Print full file
            print("\n######### File " + filename + " #########")
            with open(filename) as f:
                for index, line in enumerate(f.readlines()):
                    if index in line_updates:
                        old_version, new_version = line_updates[index]
                        if old_version in line:
                            line = line.replace(old_version, new_version)
                        else:
                            line = line.strip() + \
                                " # ERROR: could not update version string to " + \
                                new_version + "\n"
                    print(line, end="")
        if verbose:
            print("")

    return results


def verbose(filenames=None):
    """Search each file for updates. Defaults to ./requirements.txt
    Prints the package that have new versions available and the
    suggested new requirements.txt file with the new versions"""
    check_files(filenames, verbose=True)


if __name__ == "__main__":
    verbose(sys.argv[1:])
