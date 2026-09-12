#!/usr/bin/env python3
"""
code_mapper.py - Project-wide code relationship mapper for llama.cpp projects.
UnobligatedRascal

Indexes C/C++/CUDA/CMake/Python source files to extract:
- Include relationships
- Function definitions and call sites
- Macro definitions and uses
- Symbol declarations
- CMake targets, options, and flags
- Error/warning/assertion messages
- Cross-file dependencies

Usage:
    python3 code_mapper.py index /path/to/project [--project-name NAME]
    python3 code_mapper.py query <command> [args...] [--project-name NAME]
    python3 code_mapper.py trace <symbol> [--project-name NAME]
    python3 code_mapper.py deps <file-or-target> [--project-name NAME]
    python3 code_mapper.py search <pattern> [--project-name NAME]

Commands:
    index <dir>              Index a project directory
    query <cmd>              Run a query command:
        includes <file>      Files included by FILE and files that include FILE
        callers <func>       Functions that call FUNC (upstream)
        callees <func>       Functions called by FUNC (downstream)
        chain <func>         Full call chain (callers + callees, depth-limited)
        defines <symbol>     Where SYMBOL is defined
        uses <symbol>        Where SYMBOL is used
        cmake <target>       CMake target details and dependencies
        cmake-uses-flag <flag>  Targets using a CMake flag
        cmake-requires <opt>  Targets requiring a CMake option
        errors <msg>         Files containing ERROR/WARN/ASSERT with MSG
        macros <macro>       Macro definitions and expansion sites
        reverse-dep <file>   Everything that depends on FILE
        forward-dep <file>   Everything FILE depends on
        graph <file>         DOT graph of dependencies for FILE
        summary              Index statistics
    trace <symbol>           Trace full activation path for SYMBOL
    deps <target>            Show dependency tree for file or CMake target
    search <pattern>         Search across indexed content (regex)
    map <query>              Semantic map: show all related entities
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

DB_DIR = Path.home() / ".code_mapper"
DEFAULT_PROJECT = "llama_wukong"

# File type patterns
CPP_PATTERNS = {
    "cpp": [".cpp", ".cc", ".cxx", ".c++"],
    "c": [".c"],
    "cuda": [".cu", ".cuh"],
    "header": [".h", ".hpp", ".hxx", ".h++"],
    "cmake": [".cmake", "CMakeLists.txt"],
    "python": [".py"],
    "md": [".md"],
}

def get_db_path(project_name: str) -> Path:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / f"{project_name}.db"

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            type TEXT,
            size INTEGER,
            mtime REAL,
            project TEXT
        );
        
        CREATE TABLE IF NOT EXISTS includes (
            from_file TEXT,
            to_file TEXT,
            line INTEGER,
            quoted INTEGER,  -- 1=quoted include, 0=system
            PRIMARY KEY (from_file, to_file, line)
        );
        
        CREATE TABLE IF NOT EXISTS functions (
            name TEXT,
            file TEXT,
            line INTEGER,
            signature TEXT,
            is_method INTEGER,
            class_name TEXT,
            PRIMARY KEY (name, file, line)
        );
        
        CREATE TABLE IF NOT EXISTS function_calls (
            caller_name TEXT,
            caller_file TEXT,
            caller_line INTEGER,
            callee_name TEXT,
            callee_file TEXT,
            PRIMARY KEY (caller_name, caller_file, caller_line, callee_name)
        );
        
        CREATE TABLE IF NOT EXISTS macros (
            name TEXT,
            file TEXT,
            line INTEGER,
            definition TEXT,
            is_use INTEGER DEFAULT 0,
            PRIMARY KEY (name, file, line, is_use)
        );
        
        CREATE TABLE IF NOT EXISTS symbols (
            name TEXT,
            file TEXT,
            line INTEGER,
            kind TEXT,  -- variable, type, enum, namespace, template
            declaration TEXT,
            PRIMARY KEY (name, file, line, kind)
        );
        
        CREATE TABLE IF NOT EXISTS cmake_targets (
            name TEXT PRIMARY KEY,
            type TEXT,  -- executable, library, object, etc.
            source_files TEXT,  -- JSON array
            dependencies TEXT,  -- JSON array of target names
            link_libs TEXT,     -- JSON array
            compile_options TEXT, -- JSON array
            compile_definitions TEXT, -- JSON array
            include_dirs TEXT,  -- JSON array
            properties TEXT,    -- JSON object
            cmake_file TEXT,
            line INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS cmake_options (
            name TEXT PRIMARY KEY,
            value TEXT,
            description TEXT,
            cmake_file TEXT,
            line INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS cmake_flags (
            target TEXT,
            flag TEXT,
            context TEXT,  -- compile, link, install
            cmake_file TEXT,
            line INTEGER,
            PRIMARY KEY (target, flag, context, cmake_file, line)
        );
        
        CREATE TABLE IF NOT EXISTS messages (
            file TEXT,
            line INTEGER,
            type TEXT,  -- error, warn, assert, log, fatal
            message TEXT,
            function TEXT,
            PRIMARY KEY (file, line, type, message)
        );
        
        CREATE TABLE IF NOT EXISTS content_index (
            file TEXT,
            line INTEGER,
            content TEXT,
            PRIMARY KEY (file, line)
        );
        
        CREATE INDEX IF NOT EXISTS idx_includes_from ON includes(from_file);
        CREATE INDEX IF NOT EXISTS idx_includes_to ON includes(to_file);
        CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
        CREATE INDEX IF NOT EXISTS idx_function_calls_caller ON function_calls(caller_name);
        CREATE INDEX IF NOT EXISTS idx_function_calls_callee ON function_calls(callee_name);
        CREATE INDEX IF NOT EXISTS idx_macros_name ON macros(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_cmake_flags_flag ON cmake_flags(flag);
        CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(type);
        CREATE INDEX IF NOT EXISTS idx_messages_message ON messages(message);
        CREATE INDEX IF NOT EXISTS idx_content ON content_index(content);
    """)
    
    return conn

@dataclass
class FileInfo:
    path: str
    rel_path: str
    file_type: str
    content: str
    lines: list

def classify_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath).lower()
    
    # Special case: CMakeLists.txt
    if basename == 'cmakelists.txt':
        return 'cmake'
    
    for ftype, extensions in CPP_PATTERNS.items():
        if ext in extensions:
            return ftype
    return "other"

def read_file(filepath: str, max_lines: int = 50000) -> Optional[FileInfo]:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()[:max_lines]
        rel = os.path.relpath(filepath)
        return FileInfo(
            path=filepath,
            rel_path=rel,
            file_type=classify_file(filepath),
            content=''.join(lines),
            lines=lines
        )
    except:
        return None

def extract_includes(file_info: FileInfo, base_dir: str) -> list:
    includes = []
    for i, line in enumerate(file_info.lines, 1):
        match = re.match(r'\s*#\s*include\s+([<"])([^>"]+)[>"]', line.strip())
        if match:
            quoted = 1 if match.group(1) == '"' else 0
            inc_path = match.group(2)
            # Try to resolve to actual file
            resolved = None
            if quoted:
                # Relative to including file's directory
                candidate = os.path.normpath(os.path.join(
                    os.path.dirname(file_info.path), inc_path
                ))
                if os.path.exists(candidate):
                    resolved = os.path.relpath(candidate, base_dir)
            # Also try as-is relative to project root
            if not resolved:
                candidate = os.path.normpath(os.path.join(base_dir, inc_path))
                if os.path.exists(candidate):
                    resolved = os.path.relpath(candidate, base_dir)
            includes.append({
                'from_file': file_info.rel_path,
                'to_file': resolved or inc_path,
                'line': i,
                'quoted': quoted
            })
    return includes

def extract_functions(file_info: FileInfo) -> list:
    """Extract function definitions from C/C++/CUDA files."""
    if file_info.file_type not in ('cpp', 'c', 'cuda', 'header'):
        return []
    
    functions = []
    # Patterns for function definitions
    patterns = [
        # Regular function: type name(params) {
        r'^\s*(?:inline\s+|static\s+|extern\s+|virtual\s+|constexpr\s+|consteval\s+|constinit\s+)*'
        r'(?:[\w:*&<>,\s]+\s+)'
        r'(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept(?:\([^)]*\))?\s*)?'
        r'(?:->\s*[\w:*&<>,\s]+)?\s*\{',
        # Method with class scope: type Class::method(params) {
        r'^\s*(?:inline\s+|static\s+|extern\s+|virtual\s+|constexpr\s+)*'
        r'(?:[\w:*&<>,:\s]+\s+)'
        r'(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept(?:\([^)]*\))?\s*)?'
        r'(?:->\s*[\w:*&<>,\s]+)?\s*\{',
        # Lambda assignments not extracted (too complex)
    ]
    
    current_class = None
    for i, line in enumerate(file_info.lines, 1):
        stripped = line.strip()
        
        # Track class/struct/namespace context
        class_match = re.match(r'(?:class|struct)\s+(\w+)', stripped)
        if class_match:
            current_class = class_match.group(1)
        
        ns_match = re.match(r'namespace\s+(\w+)', stripped)
        if ns_match:
            current_class = None  # Reset class context on namespace
        
        for pattern in patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                name = match.group(1)
                params = match.group(2).strip() if match.group(2) else ""
                
                # Skip keywords and common false positives
                if name in ('if', 'else', 'while', 'for', 'switch', 'return', 'sizeof', 
                           'typeof', 'alignof', 'decltype', 'noexcept', 'static_assert',
                           'new', 'delete', 'throw', 'catch', 'try'):
                    continue
                
                # Check if this is a method (has :: before name)
                is_method = 0
                class_name = None
                if '::' in line.split('(')[0]:
                    scope = line.split('::')[0].strip()
                    if scope and scope[0].isalpha():
                        is_method = 1
                        class_name = scope.split()[-1]
                elif current_class and not any(kw in stripped for kw in ['class', 'struct', 'namespace']):
                    # Declaration inside class body
                    if re.match(r'.*\w+\s+' + name + r'\s*\(', stripped):
                        is_method = 1
                        class_name = current_class
                
                sig = f"{name}({params})"
                functions.append({
                    'name': name,
                    'file': file_info.rel_path,
                    'line': i,
                    'signature': sig,
                    'is_method': is_method,
                    'class_name': class_name
                })
                break
    
    return functions

def extract_function_calls(file_info: FileInfo, defined_functions: dict) -> list:
    """Extract function calls, linking to defined functions where possible."""
    if file_info.file_type not in ('cpp', 'c', 'cuda', 'header'):
        return []
    
    calls = []
    # defined_functions is dict[name] -> [func_info, ...]
    known_funcs = set(defined_functions.keys())
    
    for i, line in enumerate(file_info.lines, 1):
        stripped = line.strip()
        
        # Skip comments and includes
        if stripped.startswith('//') or stripped.startswith('#'):
            continue
        
        # Find function-like calls: name(...)
        # Be careful to avoid matching declarations
        if '{' in stripped and stripped.count('(') == stripped.count(')'):
            # Likely a definition, not a call
            continue
        
        # Find all function calls in this line
        for match in re.finditer(r'(?<!\w)(\w+)\s*\(', stripped):
            name = match.group(1)
            
            # Skip keywords
            if name in ('if', 'else', 'while', 'for', 'switch', 'return', 'sizeof',
                       'typeof', 'alignof', 'decltype', 'noexcept', 'static_assert',
                       'new', 'delete', 'throw', 'catch', 'try', 'case', 'default',
                       'sizeof', 'defined'):
                continue
            
            # Try to find which function this calls
            callee_file = None
            if name in known_funcs:
                defs = defined_functions[name]
                if defs:
                    callee_file = defs[0]['file']  # Use first definition
            
            calls.append({
                'caller_name': None,  # Will be filled in by caller context
                'caller_file': file_info.rel_path,
                'caller_line': i,
                'callee_name': name,
                'callee_file': callee_file
            })
    
    return calls

def extract_macros(file_info: FileInfo) -> list:
    """Extract macro definitions and uses."""
    macros = []
    
    for i, line in enumerate(file_info.lines, 1):
        stripped = line.strip()
        
        # Macro definitions
        def_match = re.match(r'#\s*define\s+(\w+)(?:\s*\(([^)]*)\))?\s*(.*)', stripped)
        if def_match:
            macros.append({
                'name': def_match.group(1),
                'file': file_info.rel_path,
                'line': i,
                'definition': (def_match.group(3) or '').strip(),
                'is_use': 0
            })
            continue
        
        # Macro uses (word at start of line or after #, or standalone uppercase-ish)
        # Simple heuristic: look for known macro patterns
        for match in re.finditer(r'\b(LLAMA_|GGML_|CUDA_|BUILD_|ENABLE_|USE_)\w+\b', stripped):
            macros.append({
                'name': match.group(1),
                'file': file_info.rel_path,
                'line': i,
                'definition': '',
                'is_use': 1
            })
    
    return macros

def extract_symbols(file_info: FileInfo) -> list:
    """Extract variable, type, enum, and namespace declarations."""
    if file_info.file_type not in ('cpp', 'c', 'cuda', 'header'):
        return []
    
    symbols = []
    
    for i, line in enumerate(file_info.lines, 1):
        stripped = line.strip()
        
        # Type definitions
        type_match = re.match(r'(?:typedef|using)\s+(\w+)\s*=?\s*([^;]+);', stripped)
        if type_match:
            symbols.append({
                'name': type_match.group(1),
                'file': file_info.rel_path,
                'line': i,
                'kind': 'type',
                'declaration': stripped
            })
        
        # Enum definitions
        enum_match = re.match(r'enum(?:\s+class)?\s+(\w+)', stripped)
        if enum_match:
            symbols.append({
                'name': enum_match.group(1),
                'file': file_info.rel_path,
                'line': i,
                'kind': 'enum',
                'declaration': stripped
            })
        
        # Namespace
        ns_match = re.match(r'namespace\s+(\w+)', stripped)
        if ns_match:
            symbols.append({
                'name': ns_match.group(1),
                'file': file_info.rel_path,
                'line': i,
                'kind': 'namespace',
                'declaration': stripped
            })
    
    return symbols

def extract_messages(file_info: FileInfo) -> list:
    """Extract error, warning, assert, and log messages."""
    if file_info.file_type not in ('cpp', 'c', 'cuda', 'header'):
        return []
    
    messages = []
    
    # More comprehensive patterns
    error_patterns = [
        r'LLAMA_LOG_ERROR', r'GGML_LOG_ERROR', r'log_error', r'fprintf\s*\(\s*stderr',
        r'cerr\s*<<', r'printf\s*\(\s*stderr',
    ]
    warn_patterns = [
        r'LLAMA_LOG_WARN', r'GGML_LOG_WARN', r'log_warn',
        r'(?:warning|Warning|WARN)\s*[:"(]',
    ]
    assert_patterns = [
        r'\bassert\s*\(', r'GGML_ASSERT', r'LLAMA_ASSERT', r'GGML_ASSERT_FATAL',
        r'GGML_ABORT', r'LLAMA_ABORT',
        r'(?:if\s*\([^)]*\)\s*(?:abort|exit|throw\s*(?:std::)?runtime_error))',
    ]
    fatal_patterns = [
        r'LLAMA_LOG_FATAL', r'GGML_LOG_FATAL', r'fatal_error',
    ]
    
    info_patterns = [
        r'LLAMA_LOG_INFO', r'GGML_LOG_INFO', r'log_info',
    ]
    
    def find_containing_func(start_line, max_back=150):
        for j in range(start_line - 1, max(0, start_line - max_back), -1):
            if j >= len(file_info.lines):
                continue
            func_match = re.search(r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:->\s*\w+)?\s*\{', file_info.lines[j])
            if func_match:
                name = func_match.group(1)
                if name not in ('if', 'else', 'while', 'for', 'switch', 'catch', 'try'):
                    return name
        return None
    
    def extract_and_add(line_idx, line, msg_type, patterns):
        for pat in patterns:
            if re.search(pat, line):
                msg_match = re.search(r'"([^"]{3,100})"', line)
                msg = msg_match.group(1) if msg_match else line.strip()[:100]
                func = find_containing_func(line_idx)
                messages.append({
                    'file': file_info.rel_path,
                    'line': line_idx,
                    'type': msg_type,
                    'message': msg,
                    'function': func
                })
                return True
        return False
    
    for i, line in enumerate(file_info.lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('#'):
            continue
        
        # Check each type, priority order
        if not extract_and_add(i, line, 'error', error_patterns):
            if not extract_and_add(i, line, 'fatal', fatal_patterns):
                if not extract_and_add(i, line, 'warn', warn_patterns):
                    if not extract_and_add(i, line, 'assert', assert_patterns):
                        extract_and_add(i, line, 'info', info_patterns)
    
    return messages

def extract_cmake(file_info: FileInfo, base_dir: str) -> dict:
    """Extract CMake targets, options, and flags."""
    if file_info.file_type != 'cmake':
        return {'targets': [], 'options': [], 'flags': []}
    
    targets = {}
    options = []
    flags = []
    
    content = file_info.content
    
    # CMake options - more flexible pattern
    for match in re.finditer(
        r'option\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]*)"\s+(ON|OFF|"[^"]*")\)',
        content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        val = match.group(3).strip('"')
        options.append({
            'name': match.group(1),
            'value': val,
            'description': match.group(2),
            'cmake_file': file_info.rel_path,
            'line': line_num
        })
    
    # add_executable and add_library - handle multiline with parentheses counting
    def find_cmake_args(content, start):
        """Find matching closing paren for CMake command."""
        depth = 0
        in_string = False
        string_char = None
        i = start
        while i < len(content):
            c = content[i]
            if not in_string:
                if c in ('"', "'"):
                    in_string = True
                    string_char = c
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return content[start:i], i + 1
            else:
                if c == string_char:
                    in_string = False
            i += 1
        return content[start:], len(content)
    
    # Find all add_executable/add_library/add_library calls
    for match in re.finditer(
        r'(add_(?:executable|library))\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        cmd = match.group(1).lower()
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        
        if not args:
            continue
        
        target_name = args[0]
        # Filter out keywords
        keywords = {'STATIC', 'SHARED', 'MODULE', 'OBJECT', 'IMPORTED', 'ALIAS', 'INTERFACE', 'EXCLUDE_FROM_ALL'}
        target_type = cmd.replace('add_', '')
        for a in args[1:]:
            if a.upper() in ('STATIC', 'SHARED', 'MODULE', 'OBJECT'):
                target_type = a.lower()
                break
        
        sources = [s for s in args[1:] if s not in keywords]
        
        if target_name not in targets:
            targets[target_name] = {
                'name': target_name,
                'type': target_type,
                'source_files': json.dumps(sources),
                'dependencies': json.dumps([]),
                'link_libs': json.dumps([]),
                'compile_options': json.dumps([]),
                'compile_definitions': json.dumps([]),
                'include_dirs': json.dumps([]),
                'properties': json.dumps({}),
                'cmake_file': file_info.rel_path,
                'line': line_num
            }
        else:
            # Merge sources
            existing = json.loads(targets[target_name]['source_files'])
            existing.extend(sources)
            targets[target_name]['source_files'] = json.dumps(existing)
    
    # target_link_libraries
    for match in re.finditer(r'target_link_libraries\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        
        if not args:
            continue
        
        target = args[0]
        keywords = {'PUBLIC', 'PRIVATE', 'INTERFACE'}
        libs = [l for l in args[1:] if l not in keywords]
        
        if target in targets:
            targets[target]['link_libs'] = json.dumps(libs)
            targets[target]['dependencies'] = json.dumps(libs)
    
    # target_compile_options
    for match in re.finditer(r'target_compile_options\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        
        if not args:
            continue
        
        target = args[0]
        keywords = {'PUBLIC', 'PRIVATE', 'INTERFACE'}
        opts = [o for o in args[1:] if o not in keywords]
        
        if target in targets:
            targets[target]['compile_options'] = json.dumps(opts)
        
        for opt in opts:
            flags.append({
                'target': target,
                'flag': opt,
                'context': 'compile',
                'cmake_file': file_info.rel_path,
                'line': line_num
            })
    
    # add_compile_options (global)
    for match in re.finditer(r'add_compile_options\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        args_str, _ = find_cmake_args(content, match.end())
        opts = args_str.strip().split()
        for opt in opts:
            flags.append({
                'target': '_global',
                'flag': opt,
                'context': 'compile',
                'cmake_file': file_info.rel_path,
                'line': line_num
            })
    
    # add_link_options (global)
    for match in re.finditer(r'add_link_options\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        args_str, _ = find_cmake_args(content, match.end())
        opts = args_str.strip().split()
        for opt in opts:
            flags.append({
                'target': '_global',
                'flag': opt,
                'context': 'link',
                'cmake_file': file_info.rel_path,
                'line': line_num
            })
    
    # target_compile_definitions
    for match in re.finditer(r'target_compile_definitions\s*\(', content, re.IGNORECASE):
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        
        if not args:
            continue
        
        target = args[0]
        keywords = {'PUBLIC', 'PRIVATE', 'INTERFACE'}
        defs = [d.strip('"') for d in args[1:] if d not in keywords]
        
        if target in targets:
            targets[target]['compile_definitions'] = json.dumps(defs)
    
    # add_compile_definitions (global)
    for match in re.finditer(r'add_compile_definitions\s*\(', content, re.IGNORECASE):
        args_str, _ = find_cmake_args(content, match.end())
        defs = args_str.strip().split()
        for d in defs:
            clean_d = d.strip('"').strip("'")
            flags.append({
                'target': '_global',
                'flag': f'DEF:{clean_d}',
                'context': 'compile',
                'cmake_file': file_info.rel_path,
                'line': content[:match.start()].count('\n') + 1
            })
    
    # target_include_directories
    for match in re.finditer(r'target_include_directories\s*\(', content, re.IGNORECASE):
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        
        if not args:
            continue
        
        target = args[0]
        keywords = {'PUBLIC', 'PRIVATE', 'INTERFACE', 'SYSTEM', 'BEFORE', 'AFTER'}
        dirs = [d for d in args[1:] if d not in keywords]
        
        if target in targets:
            targets[target]['include_dirs'] = json.dumps(dirs)
    
    # Custom CMake functions that create targets (ggml_add_backend_library etc.)
    for match in re.finditer(
        r'(ggml_add_backend_library|add_backend_library)\s*\(', content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        args_str, _ = find_cmake_args(content, match.end())
        args = args_str.strip().split()
        if not args:
            continue
        target_name = args[0]
        # Resolve ${VAR} references
        if target_name.startswith('${') and target_name.endswith('}'):
            var_name = target_name[2:-1]
            var_match = re.search(rf'set\s*\(\s*{re.escape(var_name)}\s+([^\)]+)\)', content)
            if var_match:
                target_name = var_match.group(1).strip().strip('"').strip("'")
            else:
                continue
        if target_name not in targets:
            targets[target_name] = {
                'name': target_name,
                'type': 'module',
                'source_files': json.dumps(args[1:]),
                'dependencies': json.dumps([]),
                'link_libs': json.dumps([]),
                'compile_options': json.dumps([]),
                'compile_definitions': json.dumps([]),
                'include_dirs': json.dumps([]),
                'properties': json.dumps({'created_by': match.group(1)}),
                'cmake_file': file_info.rel_path,
                'line': line_num
            }
    
    # set() commands for tracking important variables
    for match in re.finditer(
        r'set\s*\(\s*([A-Z_][A-Z0-9_]*)\s+([^)]+)\)', content, re.IGNORECASE):
        var = match.group(1)
        val = match.group(2).strip()
        val = val.strip('"').strip("'")
        if var.startswith('LLAMA_') or var.startswith('GGML_') or var.startswith('BUILD_'):
            if var not in targets:
                targets[var] = {
                    'name': var,
                    'type': 'variable',
                    'source_files': json.dumps([]),
                    'dependencies': json.dumps([]),
                    'link_libs': json.dumps([]),
                    'compile_options': json.dumps([]),
                    'compile_definitions': json.dumps([val]),
                    'include_dirs': json.dumps([]),
                    'properties': json.dumps({'value': val}),
                    'cmake_file': file_info.rel_path,
                    'line': content[:match.start()].count('\n') + 1
                }
    
    return {'targets': list(targets.values()), 'options': options, 'flags': flags}

def index_project(project_dir: str, project_name: str, verbose: bool = True) -> dict:
    """Index an entire project directory."""
    db_path = get_db_path(project_name)
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    start_time = time.time()
    stats = {
        'files': 0,
        'includes': 0,
        'functions': 0,
        'calls': 0,
        'macros': 0,
        'symbols': 0,
        'targets': 0,
        'options': 0,
        'flags': 0,
        'messages': 0,
        'errors': 0
    }
    
    # Clear existing index for this project
    cursor.execute("DELETE FROM files WHERE project=?", (project_name,))
    cursor.execute("DELETE FROM includes")
    cursor.execute("DELETE FROM functions")
    cursor.execute("DELETE FROM function_calls")
    cursor.execute("DELETE FROM macros")
    cursor.execute("DELETE FROM symbols")
    cursor.execute("DELETE FROM cmake_targets")
    cursor.execute("DELETE FROM cmake_options")
    cursor.execute("DELETE FROM cmake_flags")
    cursor.execute("DELETE FROM messages")
    cursor.execute("DELETE FROM content_index")
    conn.commit()
    
    # Collect all source files
    all_files = []
    for root, dirs, files in os.walk(project_dir):
        # Skip build dirs, .git, node_modules, etc.
        dirs[:] = [d for d in dirs if d not in (
            '.git', 'build', 'build_test', 'build_test2', 'node_modules',
            '__cache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache',
            'test_logs', 'test_concurrent_logs', 'bench_results',
            '.cache', 'dist', '.eggs'
        )]
        
        for fname in files:
            filepath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            # Include cmake files by extension or by name (CMakeLists.txt)
            if ext in sum(CPP_PATTERNS.values(), []) or fname.lower() == 'cmakelists.txt':
                all_files.append(filepath)
    
    if verbose:
        print(f"Found {len(all_files)} files to index...")
    
    # First pass: read all files and build function registry
    file_infos = {}
    all_functions_by_name = defaultdict(list)
    
    for filepath in all_files:
        fi = read_file(filepath)
        if not fi:
            stats['errors'] += 1
            continue
        
        file_infos[fi.rel_path] = fi
        
        # Register file
        cursor.execute(
            "INSERT OR REPLACE INTO files (path, type, size, mtime, project) VALUES (?, ?, ?, ?, ?)",
            (fi.rel_path, fi.file_type, len(fi.content), 
             os.path.getmtime(filepath), project_name)
        )
        
        # Extract functions
        funcs = extract_functions(fi)
        for f in funcs:
            all_functions_by_name[f['name']].append(f)
        
        stats['files'] += 1
    
    conn.commit()
    
    if verbose:
        print(f"Read {stats['files']} files, found {len(all_functions_by_name)} unique function names...")
    
    # Second pass: extract relationships
    for rel_path, fi in file_infos.items():
        # Includes
        includes = extract_includes(fi, project_dir)
        for inc in includes:
            cursor.execute(
                "INSERT OR IGNORE INTO includes (from_file, to_file, line, quoted) VALUES (?, ?, ?, ?)",
                (inc['from_file'], inc['to_file'], inc['line'], inc['quoted'])
            )
        stats['includes'] += len(includes)
        
        # Functions
        funcs = extract_functions(fi)
        for f in funcs:
            cursor.execute(
                "INSERT OR IGNORE INTO functions (name, file, line, signature, is_method, class_name) VALUES (?, ?, ?, ?, ?, ?)",
                (f['name'], f['file'], f['line'], f['signature'], f['is_method'], f['class_name'])
            )
        stats['functions'] += len(funcs)
        
        # Function calls (with context from defined functions in this file)
        local_funcs = {f['name']: [f] for f in funcs}
        calls = extract_function_calls(fi, local_funcs)
        
        # Now fill in caller_name by looking at context
        for call in calls:
            # Find which function contains this call
            caller = None
            for f in funcs:
                if f['line'] < call['caller_line']:
                    caller = f['name']
            call['caller_name'] = caller
            
            cursor.execute(
                "INSERT OR IGNORE INTO function_calls (caller_name, caller_file, caller_line, callee_name, callee_file) VALUES (?, ?, ?, ?, ?)",
                (call['caller_name'], call['caller_file'], call['caller_line'],
                 call['callee_name'], call['callee_file'])
            )
        stats['calls'] += len(calls)
        
        # Macros
        macros = extract_macros(fi)
        for m in macros:
            cursor.execute(
                "INSERT OR IGNORE INTO macros (name, file, line, definition, is_use) VALUES (?, ?, ?, ?, ?)",
                (m['name'], m['file'], m['line'], m['definition'], m['is_use'])
            )
        stats['macros'] += len(macros)
        
        # Symbols
        syms = extract_symbols(fi)
        for s in syms:
            cursor.execute(
                "INSERT OR IGNORE INTO symbols (name, file, line, kind, declaration) VALUES (?, ?, ?, ?, ?)",
                (s['name'], s['file'], s['line'], s['kind'], s['declaration'])
            )
        stats['symbols'] += len(syms)
        
        # Messages
        msgs = extract_messages(fi)
        for m in msgs:
            cursor.execute(
                "INSERT OR IGNORE INTO messages (file, line, type, message, function) VALUES (?, ?, ?, ?, ?)",
                (m['file'], m['line'], m['type'], m['message'], m['function'])
            )
        stats['messages'] += len(msgs)
        
        # CMake
        cmake_data = extract_cmake(fi, project_dir)
        for t in cmake_data['targets']:
            cursor.execute(
                """INSERT OR REPLACE INTO cmake_targets 
                   (name, type, source_files, dependencies, link_libs, compile_options, 
                    compile_definitions, include_dirs, properties, cmake_file, line)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (t['name'], t['type'], t['source_files'], t['dependencies'], t['link_libs'],
                 t['compile_options'], t['compile_definitions'], t['include_dirs'],
                 t['properties'], t['cmake_file'], t['line'])
            )
        stats['targets'] += len(cmake_data['targets'])
        
        for o in cmake_data['options']:
            cursor.execute(
                "INSERT OR REPLACE INTO cmake_options (name, value, description, cmake_file, line) VALUES (?, ?, ?, ?, ?)",
                (o['name'], o['value'], o['description'], o['cmake_file'], o['line'])
            )
        stats['options'] += len(cmake_data['options'])
        
        for f in cmake_data['flags']:
            cursor.execute(
                "INSERT OR IGNORE INTO cmake_flags (target, flag, context, cmake_file, line) VALUES (?, ?, ?, ?, ?)",
                (f['target'], f['flag'], f['context'], f['cmake_file'], f['line'])
            )
        stats['flags'] += len(cmake_data['flags'])
        
        # Content index for key lines
        for i, line in enumerate(fi.lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and len(stripped) < 200:
                cursor.execute(
                    "INSERT OR IGNORE INTO content_index (file, line, content) VALUES (?, ?, ?)",
                    (fi.rel_path, i, stripped)
                )
    
    conn.commit()
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nIndex complete in {elapsed:.1f}s:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  Database: {db_path}")
    
    conn.close()
    return stats

def query_db(project_name: str, query: str, params=()) -> list:
    """Execute a query and return results."""
    db_path = get_db_path(project_name)
    if not db_path.exists():
        print(f"Error: No index found for project '{project_name}'. Run 'index' first.", file=sys.stderr)
        sys.exit(1)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

def query_includes(project_name: str, file_path: str):
    """Show include relationships for a file."""
    print(f"\n=== Includes for: {file_path} ===\n")
    
    # Files this file includes
    rows = query_db(project_name, 
        "SELECT to_file, line, quoted FROM includes WHERE from_file=? ORDER BY line",
        (file_path,))
    if rows:
        print("Includes:")
        for r in rows:
            mark = "@" if r['quoted'] else ":"
            print(f"  {mark} {r['to_file']} (line {r['line']})")
    else:
        print("No includes found.")
    
    # Files that include this file
    rows = query_db(project_name,
        "SELECT from_file, line FROM includes WHERE to_file=? ORDER BY from_file",
        (file_path,))
    if rows:
        print(f"\nIncluded by ({len(rows)} files):")
        for r in rows:
            print(f"  {r['from_file']} (line {r['line']})")
    else:
        print("Not included by any indexed file.")

def query_callers(project_name: str, func_name: str, max_depth: int = 3):
    """Find functions that call the given function."""
    print(f"\n=== Callers of: {func_name} ===\n")
    
    # Direct callers
    rows = query_db(project_name,
        """SELECT DISTINCT fc.caller_name, fc.caller_file, fc.caller_line
           FROM function_calls fc
           WHERE fc.callee_name=?
           ORDER BY fc.caller_file, fc.caller_line""",
        (func_name,))
    
    if not rows:
        print("No callers found.")
        return
    
    callers = set()
    print(f"Direct callers ({len(rows)} sites):")
    for r in rows:
        print(f"  {r['caller_name']} in {r['caller_file']}:{r['caller_line']}")
        if r['caller_name']:
            callers.add(r['caller_name'])
    
    # Recursive callers
    for depth in range(2, max_depth + 1):
        if not callers:
            break
        caller_list = ','.join('?' * len(callers))
        rows = query_db(project_name,
            f"""SELECT DISTINCT fc.caller_name, fc.caller_file, fc.caller_line
               FROM function_calls fc
               WHERE fc.callee_name IN ({caller_list})
               AND fc.caller_name NOT IN ({caller_list})
               ORDER BY fc.caller_file, fc.caller_line""",
            list(callers) + list(callers))
        
        if not rows:
            break
        
        new_callers = set()
        print(f"\nDepth {depth} callers ({len(rows)} sites):")
        for r in rows:
            print(f"  {r['caller_name']} in {r['caller_file']}:{r['caller_line']}")
            if r['caller_name']:
                new_callers.add(r['caller_name'])
        callers = new_callers

def query_callees(project_name: str, func_name: str, max_depth: int = 3):
    """Find functions called by the given function."""
    print(f"\n=== Callees of: {func_name} ===\n")
    
    # Direct callees
    rows = query_db(project_name,
        """SELECT DISTINCT fc.callee_name, fc.callee_file, fc.caller_line
           FROM function_calls fc
           WHERE fc.caller_name=?
           ORDER BY fc.caller_line""",
        (func_name,))
    
    if not rows:
        print("No callees found.")
        return
    
    callees = set()
    print(f"Direct callees ({len(rows)} sites):")
    for r in rows:
        loc = f" ({r['callee_file']})" if r['callee_file'] else ""
        print(f"  {r['callee_name']}{loc} (called at line {r['caller_line']})")
        if r['callee_name']:
            callees.add(r['callee_name'])
    
    # Recursive callees
    for depth in range(2, max_depth + 1):
        if not callees:
            break
        callee_list = ','.join('?' * len(callees))
        rows = query_db(project_name,
            f"""SELECT DISTINCT fc.callee_name, fc.callee_file, fc.caller_line
               FROM function_calls fc
               WHERE fc.caller_name IN ({callee_list})
               AND fc.callee_name NOT IN ({callee_list})
               ORDER BY fc.caller_line""",
            list(callees) + list(callees))
        
        if not rows:
            break
        
        new_callees = set()
        print(f"\nDepth {depth} callees ({len(rows)} sites):")
        for r in rows:
            loc = f" ({r['callee_file']})" if r['callee_file'] else ""
            print(f"  {r['callee_name']}{loc} (called at line {r['caller_line']})")
            if r['callee_name']:
                new_callees.add(r['callee_name'])
        callees = new_callees

def query_chain(project_name: str, func_name: str, max_depth: int = 4):
    """Show full call chain around a function."""
    print(f"\n{'='*60}")
    print(f"CALL CHAIN: {func_name}")
    print(f"{'='*60}\n")
    
    # Upstream (callers)
    print("UPSTREAM (who calls this):")
    print("-" * 40)
    current = {func_name}
    visited = {func_name}
    
    for depth in range(1, max_depth + 1):
        if not current:
            break
        placeholders = ','.join('?' * len(current))
        rows = query_db(project_name,
            f"""SELECT DISTINCT caller_name FROM function_calls
               WHERE callee_name IN ({placeholders})
               AND caller_name NOT IN ({placeholders})""",
            list(current) + list(current))
        
        if not rows:
            break
        
        new_funcs = set()
        for r in rows:
            if r['caller_name'] and r['caller_name'] not in visited:
                new_funcs.add(r['caller_name'])
                visited.add(r['caller_name'])
                print(f"  {'  ' * depth}[{depth}] <- {r['caller_name']}")
        current = new_funcs
    
    print(f"\nDOWNSTREAM (what this calls):")
    print("-" * 40)
    current = {func_name}
    visited = {func_name}
    
    for depth in range(1, max_depth + 1):
        if not current:
            break
        placeholders = ','.join('?' * len(current))
        rows = query_db(project_name,
            f"""SELECT DISTINCT callee_name FROM function_calls
               WHERE caller_name IN ({placeholders})
               AND callee_name NOT IN ({placeholders})""",
            list(current) + list(current))
        
        if not rows:
            break
        
        new_funcs = set()
        for r in rows:
            if r['callee_name'] and r['callee_name'] not in visited:
                new_funcs.add(r['callee_name'])
                visited.add(r['callee_name'])
                print(f"  {'  ' * depth}[{depth}] -> {r['callee_name']}")
        current = new_funcs

def query_defines(project_name: str, symbol: str):
    """Find where a symbol is defined."""
    print(f"\n=== Definitions of: {symbol} ===\n")
    
    # Function definitions
    rows = query_db(project_name,
        "SELECT file, line, signature FROM functions WHERE name=?",
        (symbol,))
    if rows:
        print(f"Function definitions ({len(rows)}):")
        for r in rows:
            print(f"  {r['file']}:{r['line']} - {r['signature']}")
    
    # Macro definitions
    rows = query_db(project_name,
        "SELECT file, line, definition FROM macros WHERE name=? AND is_use=0",
        (symbol,))
    if rows:
        print(f"\nMacro definitions ({len(rows)}):")
        for r in rows:
            print(f"  {r['file']}:{r['line']} - {r['definition'][:80]}")
    
    # Symbol definitions
    rows = query_db(project_name,
        "SELECT file, line, kind, declaration FROM symbols WHERE name=?",
        (symbol,))
    if rows:
        print(f"\nSymbol definitions ({len(rows)}):")
        for r in rows:
            print(f"  {r['kind']}: {r['file']}:{r['line']} - {r['declaration'][:80]}")
    
    if not rows:
        print("No definitions found.")

def query_uses(project_name: str, symbol: str):
    """Find where a symbol is used."""
    print(f"\n=== Uses of: {symbol} ===\n")
    
    # Function call sites
    rows = query_db(project_name,
        """SELECT caller_name, caller_file, caller_line FROM function_calls
           WHERE callee_name=? LIMIT 50""",
        (symbol,))
    if rows:
        print(f"Called from ({len(rows)} sites, showing first 50):")
        for r in rows:
            print(f"  {r['caller_name']}@{r['caller_file']}:{r['caller_line']}")
    
    # Macro uses
    rows = query_db(project_name,
        "SELECT file, line FROM macros WHERE name=? AND is_use=1 LIMIT 50",
        (symbol,))
    if rows:
        print(f"\nMacro uses ({len(rows)} sites, showing first 50):")
        for r in rows:
            print(f"  {r['file']}:{r['line']}")

def query_cmake_target(project_name: str, target: str):
    """Show CMake target details."""
    print(f"\n=== CMake Target: {target} ===\n")
    
    rows = query_db(project_name,
        "SELECT * FROM cmake_targets WHERE name=?", (target,))
    if not rows:
        print("Target not found.")
        return
    
    t = rows[0]
    print(f"Type: {t['type']}")
    print(f"Defined in: {t['cmake_file']}:{t['line']}")
    
    sources = json.loads(t['source_files']) if t['source_files'] else []
    if sources:
        print(f"\nSources ({len(sources)}):")
        for s in sources[:20]:
            print(f"  {s}")
        if len(sources) > 20:
            print(f"  ... and {len(sources) - 20} more")
    
    deps = json.loads(t['dependencies']) if t['dependencies'] else []
    if deps:
        print(f"\nDependencies/Links ({len(deps)}):")
        for d in deps:
            print(f"  {d}")
    
    opts = json.loads(t['compile_options']) if t['compile_options'] else []
    if opts:
        print(f"\nCompile options ({len(opts)}):")
        for o in opts:
            print(f"  {o}")
    
    defs = json.loads(t['compile_definitions']) if t['compile_definitions'] else []
    if defs:
        print(f"\nDefinitions ({len(defs)}):")
        for d in defs:
            print(f"  {d}")

def query_cmake_uses_flag(project_name: str, flag: str):
    """Find targets using a CMake flag."""
    print(f"\n=== Targets using flag: {flag} ===\n")
    
    rows = query_db(project_name,
        "SELECT target, context, cmake_file, line FROM cmake_flags WHERE flag LIKE ?",
        (f'%{flag}%',))
    if not rows:
        print("No targets found with this flag.")
        return
    
    print(f"Found {len(rows)} occurrences:")
    for r in rows:
        print(f"  {r['target']} ({r['context']}) in {r['cmake_file']}:{r['line']}")

def query_cmake_requires(project_name: str, option: str):
    """Find targets/options that depend on a CMake option."""
    print(f"\n=== CMake option: {option} ===\n")
    
    # Direct option
    rows = query_db(project_name,
        "SELECT * FROM cmake_options WHERE name=?", (option,))
    if rows:
        o = rows[0]
        print(f"Value: {o['value']}")
        print(f"Description: {o['description']}")
        print(f"Defined in: {o['cmake_file']}:{o['line']}")
    
    # Targets that might use it (check definitions)
    rows = query_db(project_name,
        """SELECT name, compile_definitions FROM cmake_targets
           WHERE compile_definitions LIKE ?""",
        (f'%{option}%',))
    if rows:
        print(f"\nTargets with related definitions ({len(rows)}):")
        for r in rows:
            defs = json.loads(r['compile_definitions']) if r['compile_definitions'] else []
            related = [d for d in defs if option in d]
            if related:
                print(f"  {r['name']}: {', '.join(related)}")

def query_errors(project_name: str, msg_pattern: str):
    """Find error/warning/assert messages containing a pattern."""
    print(f"\n=== Messages matching: {msg_pattern} ===\n")
    
    rows = query_db(project_name,
        """SELECT file, line, type, message, function FROM messages
           WHERE message LIKE ? OR type LIKE ?
           ORDER BY type, file, line""",
        (f'%{msg_pattern}%', f'%{msg_pattern}%'))
    
    if not rows:
        print("No matching messages found.")
        return
    
    by_type = defaultdict(list)
    for r in rows:
        by_type[r['type']].append(r)
    
    for msg_type in ['error', 'fatal', 'warn', 'assert']:
        if msg_type in by_type:
            print(f"\n{msg_type.upper()} ({len(by_type[msg_type])}):")
            for r in by_type[msg_type]:
                func_info = f" in {r['function']}" if r['function'] else ""
                print(f"  {r['file']}:{r['line']}{func_info}")
                print(f"    {r['message'][:120]}")

def query_macros(project_name: str, macro_name: str):
    """Show macro definitions and uses."""
    print(f"\n=== Macro: {macro_name} ===\n")
    
    # Definitions
    rows = query_db(project_name,
        "SELECT file, line, definition FROM macros WHERE name=? AND is_use=0",
        (macro_name,))
    if rows:
        print("Definitions:")
        for r in rows:
            print(f"  {r['file']}:{r['line']}")
            if r['definition']:
                print(f"    {r['definition'][:100]}")
    
    # Uses
    rows = query_db(project_name,
        "SELECT file, line FROM macros WHERE name=? AND is_use=1 LIMIT 50",
        (macro_name,))
    if rows:
        print(f"\nUses ({len(rows)} sites, showing first 50):")
        for r in rows:
            print(f"  {r['file']}:{r['line']}")

def query_reverse_dep(project_name: str, file_path: str):
    """Show everything that depends on a file (transitively)."""
    print(f"\n=== Reverse dependencies of: {file_path} ===\n")
    
    # Files that directly include this file
    rows = query_db(project_name,
        "SELECT DISTINCT from_file FROM includes WHERE to_file=?",
        (file_path,))
    direct = [r['from_file'] for r in rows]
    
    if not direct:
        print("No direct dependents found.")
        return
    
    print(f"Direct dependents ({len(direct)}):")
    for f in direct:
        print(f"  {f}")
    
    # Transitive dependents (BFS)
    visited = {file_path}
    queue = list(direct)
    transitive = set()
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        transitive.add(current)
        
        rows = query_db(project_name,
            "SELECT DISTINCT from_file FROM includes WHERE to_file=?",
            (current,))
        for r in rows:
            if r['from_file'] not in visited:
                queue.append(r['from_file'])
    
    if transitive:
        print(f"\nTotal transitive dependents: {len(transitive)}")

def query_forward_dep(project_name: str, file_path: str):
    """Show everything a file depends on (transitively)."""
    print(f"\n=== Forward dependencies of: {file_path} ===\n")
    
    # Files this file directly includes
    rows = query_db(project_name,
        "SELECT DISTINCT to_file FROM includes WHERE from_file=?",
        (file_path,))
    direct = [r['to_file'] for r in rows]
    
    if not direct:
        print("No direct dependencies found.")
        return
    
    print(f"Direct dependencies ({len(direct)}):")
    for f in direct:
        print(f"  {f}")
    
    # Transitive dependencies (BFS)
    visited = {file_path}
    queue = list(direct)
    transitive = set()
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        transitive.add(current)
        
        rows = query_db(project_name,
            "SELECT DISTINCT to_file FROM includes WHERE from_file=?",
            (current,))
        for r in rows:
            if r['to_file'] not in visited:
                queue.append(r['to_file'])
    
    if transitive:
        print(f"\nTotal transitive dependencies: {len(transitive)}")

def query_graph(project_name: str, file_path: str):
    """Generate DOT graph for a file's dependencies."""
    print(f"\n=== DOT Graph for: {file_path} ===\n")
    
    print("digraph dependencies {")
    print(f'  "{file_path}";')
    
    # Direct includes
    rows = query_db(project_name,
        "SELECT to_file FROM includes WHERE from_file=?",
        (file_path,))
    
    for r in rows:
        target = r['to_file'].replace('"', '\\"')
        print(f'  "{file_path}" -> "{target}";')
        print(f'  "{target}";')
    
    # Files that include this file
    rows = query_db(project_name,
        "SELECT from_file FROM includes WHERE to_file=?",
        (file_path,))
    
    for r in rows:
        source = r['from_file'].replace('"', '\\"')
        print(f'  "{source}" -> "{file_path}";')
        print(f'  "{source}";')
    
    print("}")

def query_summary(project_name: str):
    """Show index statistics."""
    print(f"\n=== Index Summary: {project_name} ===\n")
    
    queries = [
        ("Files", "SELECT COUNT(*) as c FROM files"),
        ("Includes", "SELECT COUNT(*) as c FROM includes"),
        ("Functions", " COUNT(*) as c FROM functions"),
        ("Function calls", "SELECT COUNT(*) as c FROM function_calls"),
        ("Macros", "SELECT COUNT(*) as c FROM macros"),
        ("Symbols", "SELECT COUNT(*) as c FROM symbols"),
        ("CMake targets", "SELECT COUNT(*) as c FROM cmake_targets"),
        ("CMake options", "SELECT COUNT(*) as c FROM cmake_options"),
        ("CMake flags", "SELECT COUNT(*) as c FROM cmake_flags"),
        ("Messages", "SELECT COUNT(*) as c FROM messages"),
    ]
    
    for label, query in queries:
        rows = query_db(project_name, query)
        if rows:
            print(f"  {label}: {rows[0]['c']}")

def search_content(project_name: str, pattern: str, regex: bool = False, max_results: int = 50):
    """Search across indexed content."""
    print(f"\n=== Search: {pattern} ===\n")
    
    if regex:
        # SQLite doesn't have native regex, use LIKE with wildcards as fallback
        sql_pattern = pattern.replace('*', '%').replace('?', '_')
        rows = query_db(project_name,
            f"""SELECT file, line, content FROM content_index
               WHERE content LIKE ? LIMIT ?""",
            (sql_pattern, max_results))
    else:
        rows = query_db(project_name,
            """SELECT file, line, content FROM content_index
               WHERE content LIKE ? LIMIT ?""",
            (f'%{pattern}%', max_results))
    
    if not rows:
        print("No results found.")
        return
    
    print(f"Found {len(rows)} results:")
    for r in rows:
        # Highlight match
        content = r['content']
        print(f"  {r['file']}:{r['line']}")
        print(f"    {content[:150]}")

def map_entities(project_name: str, query: str):
    """Semantic map: show all related entities for a query."""
    print(f"\n{'='*60}")
    print(f"ENTITY MAP: {query}")
    print(f"{'='*60}\n")
    
    # Search functions
    funcs = query_db(project_name,
        "SELECT name, file, line FROM functions WHERE name LIKE ? LIMIT 10",
        (f'%{query}%',))
    if funcs:
        print(f"Functions ({len(funcs)}):")
        for f in funcs:
            print(f"  {f['name']} @ {f['file']}:{f['line']}")
    
    # Search macros
    macros = query_db(project_name,
        "SELECT name, file, line FROM macros WHERE name LIKE ? AND is_use=0 LIMIT 10",
        (f'%{query}%',))
    if macros:
        print(f"\nMacros ({len(macros)}):")
        for m in macros:
            print(f"  {m['name']} @ {m['file']}:{m['line']}")
    
    # Search symbols
    syms = query_db(project_name,
        "SELECT name, file, line, kind FROM symbols WHERE name LIKE ? LIMIT 10",
        (f'%{query}%',))
    if syms:
        print(f"\nSymbols ({len(syms)}):")
        for s in syms:
            print(f"  [{s['kind']}] {s['name']} @ {s['file']}:{s['line']}")
    
    # Search CMake targets
    targets = query_db(project_name,
        "SELECT name, type FROM cmake_targets WHERE name LIKE ? LIMIT 10",
        (f'%{query}%',))
    if targets:
        print(f"\nCMake targets ({len(targets)}):")
        for t in targets:
            print(f"  [{t['type']}] {t['name']}")
    
    # Search messages
    msgs = query_db(project_name,
        "SELECT type, file, line FROM messages WHERE message LIKE ? LIMIT 10",
        (f'%{query}%',))
    if msgs:
        print(f"\nMessages ({len(msgs)}):")
        for m in msgs:
            print(f"  [{m['type']}] {m['file']}:{m['line']}")
    
    # Search files
    files = query_db(project_name,
        "SELECT path, type FROM files WHERE path LIKE ? LIMIT 10",
        (f'%{query}%',))
    if files:
        print(f"\nFiles ({len(files)}):")
        for f in files:
            print(f"  [{f['type']}] {f['path']}")

def main():
    parser = argparse.ArgumentParser(
        description='code_mapper - Project code relationship mapper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--project-name', '-p', default=DEFAULT_PROJECT,
                       help=f'Project name for index (default: {DEFAULT_PROJECT})')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # index command
    idx_parser = subparsers.add_parser('index', help='Index a project')
    idx_parser.add_argument('directory', help='Project root directory')
    idx_parser.add_argument('--quiet', '-q', action='store_true')
    
    # query command
    qry_parser = subparsers.add_parser('query', help='Run a query')
    qry_parser.add_argument('subcommand', choices=[
        'includes', 'callers', 'callees', 'chain', 'defines', 'uses',
        'cmake', 'cmake-uses-flag', 'cmake-requires', 'errors', 'macros',
        'reverse-dep', 'forward-dep', 'graph', 'summary'
    ])
    qry_parser.add_argument('args', nargs='*', help='Query arguments')
    qry_parser.add_argument('--depth', '-d', type=int, default=3,
                           help='Depth for recursive queries')
    
    # trace command
    trace_parser = subparsers.add_parser('trace', help='Trace activation path')
    trace_parser.add_argument('symbol', help='Symbol to trace')
    trace_parser.add_argument('--depth', '-d', type=int, default=4)
    
    # deps command
    deps_parser = subparsers.add_parser('deps', help='Show dependencies')
    deps_parser.add_argument('target', help='File or CMake target')
    
    # search command
    search_parser = subparsers.add_parser('search', help='Search content')
    search_parser.add_argument('pattern', help='Search pattern')
    search_parser.add_argument('--regex', '-r', action='store_true')
    search_parser.add_argument('--max', '-m', type=int, default=50)
    
    # map command
    map_parser = subparsers.add_parser('map', help='Map related entities')
    map_parser.add_argument('query', help='Query term')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        index_project(args.directory, args.project_name, verbose=not args.quiet)
    
    elif args.command == 'query':
        if args.subcommand == 'includes':
            query_includes(args.project_name, args.args[0])
        elif args.subcommand == 'callers':
            query_callers(args.project_name, args.args[0], args.depth)
        elif args.subcommand == 'callees':
            query_callees(args.project_name, args.args[0], args.depth)
        elif args.subcommand == 'chain':
            query_chain(args.project_name, args.args[0], args.depth)
        elif args.subcommand == 'defines':
            query_defines(args.project_name, args.args[0])
        elif args.subcommand == 'uses':
            query_uses(args.project_name, args.args[0])
        elif args.subcommand == 'cmake':
            query_cmake_target(args.project_name, args.args[0])
        elif args.subcommand == 'cmake-uses-flag':
            query_cmake_uses_flag(args.project_name, args.args[0])
        elif args.subcommand == 'cmake-requires':
            query_cmake_requires(args.project_name, args.args[0])
        elif args.subcommand == 'errors':
            query_errors(args.project_name, args.args[0])
        elif args.subcommand == 'macros':
            query_macros(args.project_name, args.args[0])
        elif args.subcommand == 'reverse-dep':
            query_reverse_dep(args.project_name, args.args[0])
        elif args.subcommand == 'forward-dep':
            query_forward_dep(args.project_name, args.args[0])
        elif args.subcommand == 'graph':
            query_graph(args.project_name, args.args[0])
        elif args.subcommand == 'summary':
            query_summary(args.project_name)
    
    elif args.command == 'trace':
        query_chain(args.project_name, args.symbol, args.depth)
    
    elif args.command == 'deps':
        # Try as file first, then CMake target
        rows = query_db(args.project_name, "SELECT path FROM files WHERE path=?", (args.target,))
        if rows:
            query_forward_dep(args.project_name, args.target)
        else:
            query_cmake_target(args.project_name, args.target)
    
    elif args.command == 'search':
        search_content(args.project_name, args.pattern, args.regex, args.max)
    
    elif args.command == 'map':
        map_entities(args.project_name, args.query)

if __name__ == '__main__':
    main()
