"""Generated Python execution and sample-based tool evidence."""
from typing import Any, Dict, Optional, Tuple
import ast
import os
import pickle
import re
import subprocess
import sys
import tempfile

import pandas as pd


# ── 白名单 / 黑名单定义 ──────────────────────────────────────────────

ALLOWED_MODULES = {
    # 数据操作
    'pandas', 'numpy', 'math', 'decimal', 'statistics',
    # 字符串 / 文本
    're', 'string', 'difflib', 'unicodedata',
    # 集合 / 工具
    'collections', 'itertools', 'functools', 'operator',
    # 时间
    'datetime', 'dateutil',
    # 字符串相似度
    'rapidfuzz', 'jellyfish', 'Levenshtein',
    # 序列化 / 类型
    'json', 'csv', 'hashlib', 'copy', 'typing',
    # pickle 仅在 preamble 中用于加载数据，用户代码可能间接触发
    'pickle', '_pickle',
    # pandas / numpy 内部依赖（避免间接 import 失败）
    'numbers', 'struct', 'io', 'warnings', 'textwrap',
    'abc', 'contextlib', 'enum', 'dataclasses',
}

BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'signal',
    'ctypes', 'socket', 'http', 'urllib', 'requests', 'httpx',
    'pip', 'setuptools', 'pkg_resources',
    'multiprocessing', 'threading',
}


# ── Preamble 模板 ────────────────────────────────────────────────────

_PREAMBLE_TEMPLATE = r'''
# === sandbox preamble (auto-generated, do not edit) ===
import sys as _sys

_ALLOWED_MODULES = {allowed_modules_repr}
_BLOCKED_MODULES = {blocked_modules_repr}

class _ImportWhitelist:
    """只允许白名单模块被导入；黑名单模块给出清晰提示。"""
    def find_module(self, name, path=None):
        top = name.split('.')[0]
        if top in _BLOCKED_MODULES:
            return self
        if top not in _ALLOWED_MODULES:
            return self
        return None  # 放行，由正常 import 机制处理

    def load_module(self, name):
        top = name.split('.')[0]
        allowed_str = ', '.join(sorted(_ALLOWED_MODULES - {{'pickle', '_pickle', 'numbers',
            'struct', 'io', 'warnings', 'textwrap', 'abc', 'contextlib', 'enum', 'dataclasses'}}))
        raise ImportError(
            f"Module '{{name}}' is not available in this sandbox.\n"
            f"Allowed modules: {{allowed_str}}"
        )

_sys.meta_path.insert(0, _ImportWhitelist())

# --- 加载数据 ---
import pickle as _pickle
with open(r'{data_path}', 'rb') as _f:
    _data = _pickle.load(_f)
source_df = _data['source_df']
target_df = _data['target_df']
src_col = _data['src_col']
target_cols = list(target_df.columns)
del _pickle, _f, _data

import pandas as pd
import numpy as np
import re
import collections
from collections import Counter, defaultdict

# === 跨表辅助函数（Schema Matching 专用）===

def value_overlap(col_a, col_b, normalize=True):
    """
    计算两个 Series 之间的值重叠。

    Args:
        col_a: pandas Series（源列或目标列）
        col_b: pandas Series（源列或目标列）
        normalize: 是否将值转为小写字符串再比较

    Returns:
        dict: {{"overlap": int, "jaccard": float, "src_ratio": float, "tgt_ratio": float}}
    """
    if normalize:
        a = set(col_a.dropna().astype(str).str.strip().str.lower())
        b = set(col_b.dropna().astype(str).str.strip().str.lower())
    else:
        a = set(col_a.dropna().astype(str))
        b = set(col_b.dropna().astype(str))
    inter = len(a & b)
    union = len(a | b)
    return {{
        "overlap": inter,
        "jaccard": inter / union if union else 0.0,
        "src_ratio": inter / len(a) if a else 0.0,
        "tgt_ratio": inter / len(b) if b else 0.0,
    }}

class _HelperRecord(dict):
    """Read aliases without duplicating keys in printed/serialized evidence."""
    def __init__(self, values, aliases):
        super().__init__(values)
        self._aliases = aliases

    def __missing__(self, key):
        canonical = self._aliases.get(key)
        if canonical is not None and dict.__contains__(self, canonical):
            return dict.__getitem__(self, canonical)
        if self._aliases.get("samples") == "sample" and key in ("sample", "samples"):
            return []  # Read default for all-null profiles; keep printed keys unchanged.
        if self._aliases.get("samples") == "sample" and key == "fill_rate":
            total = dict.__getitem__(self, "total")
            return dict.__getitem__(self, "non_null") / total if total else 0.0
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

def compare_all_targets(src_series, target_df, metric="jaccard", top_k=10, normalize=True):
    """
    将 source 列与 target 表的所有列比较，返回 top_k 最相似的列。

    Args:
        src_series: pandas Series（源列）
        target_df: pandas DataFrame（目标表）
        metric: 排序指标，可选 "jaccard", "overlap", "src_ratio", "tgt_ratio"
        top_k: 返回前几名
        normalize: 是否归一化比较

    Returns:
        list[dict]: 每个 dict 包含 target_column 和各指标
    """
    results = []
    for col in target_df.columns:
        r = value_overlap(src_series, target_df[col], normalize=normalize)
        r["target_column"] = col
        r["target_dtype"] = str(target_df[col].dtype)
        results.append(_HelperRecord(r, {{"column": "target_column"}}))
    results.sort(key=lambda x: x.get(metric, 0), reverse=True)
    return results[:top_k]

def column_profile(series, sample_size=10):
    """
    生成列的摘要特征。

    Returns:
        dict: 包含 dtype, non_null, null, unique, sample, top_values 等字段
    """
    profile = _HelperRecord({{
        "dtype": str(series.dtype),
        "total": len(series),
        "non_null": int(series.notna().sum()),
        "null": int(series.isna().sum()),
        "unique": int(series.nunique()),
    }}, {{"non_null_count": "non_null", "null_count": "null",
         "unique_count": "unique", "samples": "sample"}})
    non_null = series.dropna()
    if len(non_null) > 0:
        profile["sample"] = non_null.sample(min(sample_size, len(non_null)), random_state=42).tolist()
        top = non_null.astype(str).value_counts().head(5)
        profile["top_values"] = {{str(k): int(v) for k, v in top.items()}}
        str_vals = non_null.astype(str)
        lengths = str_vals.str.len()
        profile["avg_length"] = round(float(lengths.mean()), 1)
        profile["min_length"] = int(lengths.min())
        profile["max_length"] = int(lengths.max())
    return profile

def profile_all_columns(df, sample_size=5):
    """
    对 DataFrame 的所有列生成摘要。

    Returns:
        dict: {{column_name: profile_dict, ...}}
    """
    return {{col: column_profile(df[col], sample_size) for col in df.columns}}

# === sandbox preamble end ===
'''


# ── 执行器类 ─────────────────────────────────────────────────────────

class CodeExecutor:
    """
    子进程沙箱代码执行器

    安全措施：
    - 在独立子进程中执行（主进程不受影响）
    - import 白名单（只允许数据分析相关库）
    - 超时限制
    - 输出长度限制
    - 禁止文件写入 / 网络 / 系统调用
    """

    def __init__(
        self,
        max_output_length: int = 5000,
        timeout: int = 30,
        workspace_dir: Optional[str] = None,
    ):
        """
        初始化执行器

        Args:
            max_output_length: 最大输出长度（字符）
            timeout: 执行超时（秒）
            workspace_dir: 工作目录（用于存放临时文件）
        """
        self.max_output_length = max_output_length
        self.timeout = timeout
        if workspace_dir is None:
            workspace_dir = tempfile.gettempdir()
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _build_preamble(self, data_path: str) -> str:
        """生成子进程脚本的前置代码（import 白名单 + 数据加载 + 辅助函数）"""
        return _PREAMBLE_TEMPLATE.format(
            allowed_modules_repr=repr(ALLOWED_MODULES),
            blocked_modules_repr=repr(BLOCKED_MODULES),
            data_path=data_path.replace('\\', '\\\\'),
        )

    def execute(
        self,
        code: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_col: str,
    ) -> Dict[str, Any]:
        """
        在子进程沙箱中执行 Python 代码

        Args:
            code: 要执行的代码
            source_df: 源表 DataFrame
            target_df: 目标表 DataFrame
            source_col: 当前源列名

        Returns:
            {"success": bool, "output": str, "error": str|None}
        """
        if not isinstance(code, str):
            return {'success': False, 'output': '', 'error': 'Python code must be a string.'}
        validation_error = self.validate_code(code)
        if validation_error:
            return {'success': False, 'output': '', 'error': validation_error}
        data_path = None
        script_path = None

        try:
            # 1. 序列化数据到临时 pickle 文件
            fd_data, data_path = tempfile.mkstemp(
                suffix='.pkl', dir=self.workspace_dir
            )
            os.close(fd_data)
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'source_df': source_df,
                    'target_df': target_df,
                    'src_col': source_col,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 2. 构建完整脚本 = preamble + 用户代码
            preamble = self._build_preamble(data_path)
            full_script = preamble + '\n' + code

            # 3. 写入临时脚本文件
            fd_script, script_path = tempfile.mkstemp(
                suffix='.py', dir=self.workspace_dir
            )
            os.close(fd_script)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(full_script)

            # 4. 在子进程中执行
            child_env = {key: os.environ[key] for key in
                         ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
            child_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # 禁止 GUI 弹窗
            child_env.setdefault('MPLBACKEND', 'Agg')
            child_env.setdefault('QT_QPA_PLATFORM', 'offscreen')
            child_env.pop('DISPLAY', None)

            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=child_env,
            )

            stdout = completed.stdout or ''
            stderr = completed.stderr or ''

            # 5. 解析结果
            if completed.returncode == 0:
                output = stdout.strip()
                if not output:
                    output = '(无输出)'
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + '\n... [输出被截断]'
                return {
                    'success': True,
                    'output': output,
                    'error': None,
                }
            else:
                # 解析错误信息
                error_info = self._parse_error(
                    stderr, code, preamble_lines=preamble.count('\n') + 1,
                    script_path=script_path)
                partial_output = stdout.strip()
                if len(partial_output) > self.max_output_length:
                    partial_output = partial_output[:self.max_output_length] + '\n... [输出被截断]'
                return {
                    'success': False,
                    'output': partial_output,
                    'error': error_info,
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'代码执行超时（>{self.timeout}秒）。请简化代码或减少计算量。',
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'执行器内部错误: {type(e).__name__}。请检查执行环境。',
            }
        finally:
            # 清理临时文件
            for path in (data_path, script_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    def _parse_error(self, stderr: str, user_code: str, *,
                     preamble_lines: Optional[int] = None, script_path: Optional[str] = None) -> str:
        """
        从子进程 stderr 中解析出结构化错误信息

        参考 DeepAnalyze/deepanalyze.py 的错误格式化思路：
        定位到出错行号和代码行，给模型更好的修复信号。
        """
        lines = stderr.strip().splitlines()
        if not lines:
            return '未知错误（无 stderr 输出）'

        # 提取最后的错误类型和消息（通常是最后一行）
        error_line = lines[-1].strip() if lines else ''

        # 尝试从 traceback 中找到用户代码的出错行
        # execute() supplies the actual rendered prefix length. Ignore frames
        # from pandas/other modules rather than treating them as user code.
        if preamble_lines is None:
            preamble_lines = _PREAMBLE_TEMPLATE.count('\n') + 1
        user_code_lines = user_code.splitlines()

        user_error_line_num = None
        user_error_line_content = None

        for line in lines:
            if 'File' in line and 'line' in line:
                try:
                    # 提取行号
                    match = re.search(r'File "([^"]+)", line\s+(\d+)', line)
                    if match:
                        if script_path is not None and match.group(1) != script_path:
                            continue
                        abs_line = int(match.group(2))
                        # 转换为用户代码中的行号
                        rel_line = abs_line - preamble_lines
                        if 1 <= rel_line <= len(user_code_lines):
                            user_error_line_num = rel_line
                            user_error_line_content = user_code_lines[rel_line - 1].strip()
                except (ValueError, IndexError):
                    pass

        # 构建格式化错误信息
        parts = []
        parts.append(error_line)

        if user_error_line_num and user_error_line_content:
            parts.append(f'出错位置: 第 {user_error_line_num} 行: {user_error_line_content}')

        # 对常见错误类型添加针对性提示
        if 'ImportError' in error_line or 'ModuleNotFoundError' in error_line:
            allowed = sorted(ALLOWED_MODULES - {
                'pickle', '_pickle', 'numbers', 'struct', 'io',
                'warnings', 'textwrap', 'abc', 'contextlib', 'enum', 'dataclasses',
            })
            parts.append(f'可用模块: {", ".join(allowed)}')
        elif 'NameError' in error_line:
            parts.append(
                '预置变量: source_df, target_df, src_col, target_cols, pd, np, re, collections, Counter, defaultdict。'
                '预置函数: value_overlap(), compare_all_targets(), column_profile(), profile_all_columns()'
            )
        elif ('ValueError' in error_line and
              ('could not convert string to float' in error_line or 'invalid literal for int' in error_line)):
            parts.append(
                "For numeric profiling, use numeric = pd.to_numeric(series, errors='coerce'). "
                "Report invalid = series.notna() & numeric.isna(), its count and original values; "
                "compute ranges only on numeric.dropna(). Do not silently discard invalid values "
                "or overwrite source_df/target_df."
            )

        return '\n'.join(parts)

    @staticmethod
    def _compile_check_text(code: str) -> str:
        """Exempt calls to unshadowed re.compile from the legacy compile check.

        Only the validation copy is edited; executed code is unchanged. Keep
        conservative legacy rejection for aliases, rebinding and invalid syntax.
        This is a targeted false-positive fix, not a complete sandbox boundary.
        """
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return code
        nodes = list(ast.walk(tree))
        for node in nodes:
            if isinstance(node, ast.Name) and node.id == 're' and not isinstance(node.ctx, ast.Load):
                return code
            if isinstance(node, ast.arg) and node.arg == 're':
                return code
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)) and node.name == 're':
                return code
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound = alias.asname or alias.name.split('.')[0]
                    if bound == 're' and not (isinstance(node, ast.Import) and alias.name == 're'):
                        return code
                    if isinstance(node, ast.ImportFrom) and alias.name == '*':
                        return code
            # Includes re.compile assignment and re.__dict__ item assignment.
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if any(isinstance(part, ast.Name) and part.id == 're' for part in ast.walk(node)):
                    return code
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'setattr', 'delattr'}:
                return code
            if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == 're':
                return code
            if isinstance(node, ast.MatchMapping) and node.rest == 're':
                return code
        lines = code.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode('utf-8')))
        checked = bytearray(code.encode('utf-8'))
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == 'compile'
                    and isinstance(func.value, ast.Name) and func.value.id == 're'):
                end = offsets[func.end_lineno - 1] + func.end_col_offset
                checked[end - len('compile'):end] = b'REGEX__'
        return checked.decode('utf-8')

    def validate_code(self, code: str) -> Optional[str]:
        """
        执行前的静态代码检查

        两层防御：
        1. 这里拦截危险的 import 和函数调用（第一道防线）
           —— 因为 os/subprocess 等作为 pandas 的依赖已在 sys.modules 中，
              运行时 import hook 无法拦截，必须在执行前静态检查。
        2. 子进程的 import 白名单（第二道防线）拦截其他未知模块。

        Returns:
            如果有问题返回错误信息，否则返回 None
        """
        # Data is already supplied in memory. Block file/network-capable entry
        # points even when a function is assigned to an alias before calling it.
        # This is defense in depth, not operating-system isolation.
        io_names = {
            'open', 'read_csv', 'read_table', 'read_fwf', 'read_json', 'read_html',
            'read_xml', 'read_excel', 'ExcelFile', 'ExcelWriter', 'read_parquet',
            'read_feather', 'read_orc', 'read_pickle', 'read_hdf', 'HDFStore',
            'read_sql', 'read_sql_query', 'read_sql_table', 'read_sas', 'read_spss',
            'read_stata', 'read_clipboard', 'to_csv', 'to_json', 'to_html',
            'to_xml', 'to_excel', 'to_parquet', 'to_feather', 'to_orc',
            'to_pickle', 'to_hdf', 'to_sql', 'to_stata', 'to_clipboard',
            'load', 'loads', 'dump', 'save', 'savez', 'savez_compressed',
            'loadtxt', 'savetxt', 'genfromtxt', 'fromfile', 'tofile', 'memmap',
            'DataSource', 'ctypeslib', 'open_memmap', 'get_handle', 'urlopen',
            'urlretrieve', 'system', 'popen', 'fork', 'spawn',
        }
        try:
            nodes = list(ast.walk(ast.parse(code)))
        except (SyntaxError, ValueError):
            return 'Invalid Python syntax.'
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ''])
                if any(m.split('.')[0] not in ALLOWED_MODULES for m in modules):
                    return 'Only the documented data-analysis imports are allowed.'
                if any(a.name in {'pickle', '_pickle'} for a in node.names) or any(
                        m.split('.')[0] in {'pickle', '_pickle'} for m in modules):
                    return 'Deserialization is not available to generated programs.'
            # JSON in-memory parsing is permitted; pickle.loads is not.
            if isinstance(node, ast.Attribute) and node.attr in io_names - {'loads'}:
                return 'File and external I/O are unavailable; use source_df and target_df.'
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_MODULES | {'io', 'modules', 'meta_path'}:
                return 'System and network access are unavailable to generated programs.'
            if isinstance(node, ast.Name) and node.id in io_names - {'loads'}:
                return 'File and external I/O are unavailable; use source_df and target_df.'
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return 'Runtime introspection is unavailable to generated programs.'
            if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in {
                    '_sys', '_pickle', '_ImportWhitelist', '_ALLOWED_MODULES', '_BLOCKED_MODULES',
                    'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
                    'eval', 'exec', 'compile', '__import__'}):
                return 'Runtime introspection and dynamic code execution are unavailable.'

        # 危险函数调用
        forbidden_calls = [
            (r'\beval\s*\(', '禁止使用 eval() 函数'),
            (r'\bexec\s*\(', '禁止使用 exec() 函数'),
            (r'\b__import__\s*\(', '禁止使用 __import__() 函数'),
            (r'\bcompile\s*\(', '禁止使用 compile() 函数'),
            (r'\bopen\s*\(', '禁止使用 open() 函数（数据已通过 source_df/target_df 预加载）'),
        ]

        for pattern, message in forbidden_calls:
            checked = self._compile_check_text(code) if pattern == r'\bcompile\s*\(' else code
            if re.search(pattern, checked):
                return message

        # 危险模块导入（这些模块作为 pandas/numpy 的依赖已在 sys.modules 中，
        # import hook 无法拦截，必须在此静态拦截）
        forbidden_imports = [
            (r'\bimport\s+os\b', 'os'),
            (r'\bfrom\s+os\b', 'os'),
            (r'\bimport\s+sys\b', 'sys'),
            (r'\bfrom\s+sys\b', 'sys'),
            (r'\bimport\s+subprocess\b', 'subprocess'),
            (r'\bfrom\s+subprocess\b', 'subprocess'),
            (r'\bimport\s+shutil\b', 'shutil'),
            (r'\bfrom\s+shutil\b', 'shutil'),
            (r'\bimport\s+pathlib\b', 'pathlib'),
            (r'\bfrom\s+pathlib\b', 'pathlib'),
            (r'\bimport\s+signal\b', 'signal'),
            (r'\bfrom\s+signal\b', 'signal'),
            (r'\bimport\s+socket\b', 'socket'),
            (r'\bfrom\s+socket\b', 'socket'),
            (r'\bimport\s+requests\b', 'requests'),
            (r'\bfrom\s+requests\b', 'requests'),
            (r'\bimport\s+urllib\b', 'urllib'),
            (r'\bfrom\s+urllib\b', 'urllib'),
            (r'\bimport\s+http\b', 'http'),
            (r'\bfrom\s+http\b', 'http'),
            (r'\bimport\s+ctypes\b', 'ctypes'),
            (r'\bfrom\s+ctypes\b', 'ctypes'),
        ]

        allowed_str = ', '.join(sorted(
            ALLOWED_MODULES - {
                'pickle', '_pickle', 'numbers', 'struct', 'io',
                'warnings', 'textwrap', 'abc', 'contextlib', 'enum', 'dataclasses',
            }
        ))

        for pattern, mod_name in forbidden_imports:
            if re.search(pattern, code):
                return (
                    f"禁止导入 {mod_name} 模块。"
                    f"可用模块: {allowed_str}"
                )

        return None


PYTHON_EVIDENCE_SCOPE = (
    "These statistics describe the supplied samples, not whether a schema correspondence exists. "
    "Empty samples or zero fill rate alone neither establish nor refute a match. "
    "For columns present in target_df, assess names, definitions and semantic roles; "
    "do not discard a semantically supported candidate solely for missing values. "
    "An absent column or an explicit semantic conflict can still justify rejection."
)


def execute_python(
    executor: CodeExecutor,
    code: str,
    source_df: Optional[pd.DataFrame],
    target_df: Optional[pd.DataFrame],
    source_column: Optional[str],
    calls_used: int,
    max_calls: Optional[int],
) -> Tuple[Dict[str, Any], int]:
    if source_df is None or target_df is None:
        return {"error": "No task loaded"}, calls_used

    if max_calls is not None and calls_used >= max_calls:
        return {
            "success": False,
            "error": (
                f"execute_python budget exhausted for this task "
                f"({calls_used}/{max_calls}). "
                "Use shortlist_match, differential_verify, get_sample_data, doc_query, or submit."
            ),
        }, calls_used

    if not isinstance(code, str):
        return {
            "success": False,
            "error_kind": "invalid_tool_arguments",
            "error": 'execute_python requires a string code argument containing Python source. Send one JSON object {"code": "..."}.',
        }, calls_used

    try:
        ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        # The legacy parser preserves malformed JSON as raw code. Do not
        # guess at mixed JSON/Python escaping or execute the envelope itself.
        looks_like_envelope = code.lstrip().startswith('{"code"')
        message = (
            'Malformed execute_python JSON was received as Python code. '
            'Resend ONE valid JSON object with a string code field; do not nest another code object. '
            'Encode Python line breaks as \\n in JSON. A literal Python backslash must be escaped as \\\\ in JSON. '
            'No code ran and no successful-call budget was consumed.'
            if looks_like_envelope else
            f'Python syntax error in your code at line {getattr(exc, "lineno", None)}: '
            f'{getattr(exc, "msg", str(exc))}. Correct the code and resend a valid JSON code argument. '
            'No code ran and no successful-call budget was consumed.'
        )
        return {"success": False, "error_kind": "invalid_tool_arguments" if looks_like_envelope else "python_syntax_error",
                "error": message}, calls_used

    # Diagnose invalid envelopes/syntax before inspecting Python operations.
    validation_error = executor.validate_code(code)
    if validation_error:
        return {"success": False, "error": validation_error}, calls_used

    result = executor.execute(
        code=code,
        source_df=source_df,
        target_df=target_df,
        source_col=source_column,
    )
    if result.get("output"):
        result["evidence_scope"] = PYTHON_EVIDENCE_SCOPE
    if result.get("success", True):
        calls_used += 1
        result["execute_python_budget"] = {
            "used": calls_used,
            "max": max_calls,
            "remaining": None if max_calls is None else max(0, max_calls - calls_used),
        }
    return result, calls_used
