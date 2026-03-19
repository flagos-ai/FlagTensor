import hashlib
import inspect
import json
import math
import os
import sqlite3
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import triton

from flagtensor.runtime import device, torch_device_fn


def _config_cache_dir() -> Path:
    path = Path.home() / ".cache" / "flagtensor"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _triton_version_parts() -> Tuple[str, str]:
    version = getattr(triton, "__version__", "0.0")
    parts = version.split(".")
    major = parts[0] if len(parts) > 0 else "0"
    minor = parts[1] if len(parts) > 1 else "0"
    return major, minor


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_tuple_key(key: Tuple[Any, ...]) -> str:
    return json.dumps([_serialize_scalar(v) for v in key], separators=(",", ":"))


def _config_all_kwargs(config: triton.Config) -> Dict[str, Any]:
    if hasattr(config, "all_kwargs"):
        return config.all_kwargs()
    data = dict(getattr(config, "kwargs", {}))
    data["num_warps"] = getattr(config, "num_warps", None)
    data["num_stages"] = getattr(config, "num_stages", None)
    if hasattr(config, "num_ctas"):
        data["num_ctas"] = getattr(config, "num_ctas")
    return data


def _serialize_config(config: triton.Config) -> str:
    payload = {
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", 1),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _deserialize_config(payload: str) -> triton.Config:
    data = json.loads(payload)
    kwargs = data.get("kwargs", {})
    return triton.Config(
        kwargs,
        num_warps=data.get("num_warps"),
        num_stages=data.get("num_stages"),
        num_ctas=data.get("num_ctas", 1),
    )


class SQLCacheStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config_cache (
                table_name TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                config_payload TEXT NOT NULL,
                PRIMARY KEY (table_name, cache_key)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_cache (
                table_name TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                config_key TEXT NOT NULL,
                timings_payload TEXT NOT NULL,
                PRIMARY KEY (table_name, cache_key, config_key)
            )
            """
        )
        self.conn.commit()

    def get_config(self, table_name: str, cache_key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT config_payload FROM config_cache WHERE table_name = ? AND cache_key = ?",
            (table_name, cache_key),
        ).fetchone()
        return row[0] if row else None

    def set_config(self, table_name: str, cache_key: str, payload: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO config_cache(table_name, cache_key, config_payload) VALUES (?, ?, ?)",
            (table_name, cache_key, payload),
        )
        self.conn.commit()

    def get_benchmark(self, table_name: str, cache_key: str, config_key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT timings_payload FROM benchmark_cache WHERE table_name = ? AND cache_key = ? AND config_key = ?",
            (table_name, cache_key, config_key),
        ).fetchone()
        return row[0] if row else None

    def set_benchmark(self, table_name: str, cache_key: str, config_key: str, payload: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO benchmark_cache(table_name, cache_key, config_key, timings_payload) VALUES (?, ?, ?, ?)",
            (table_name, cache_key, config_key, payload),
        )
        self.conn.commit()


class ConfigCache:
    def __init__(self, table: str, store: SQLCacheStore):
        self.table = table
        self.store = store
        self.memory: Dict[Tuple[Any, ...], triton.Config] = {}

    def __contains__(self, key: Tuple[Any, ...]) -> bool:
        if key in self.memory:
            return True
        payload = self.store.get_config(self.table, _serialize_tuple_key(key))
        if payload is None:
            return False
        self.memory[key] = _deserialize_config(payload)
        return True

    def __getitem__(self, key: Tuple[Any, ...]) -> triton.Config:
        if key not in self and key not in self.memory:
            raise KeyError(key)
        return self.memory[key]

    def __setitem__(self, key: Tuple[Any, ...], value: triton.Config):
        self.memory[key] = value
        self.store.set_config(self.table, _serialize_tuple_key(key), _serialize_config(value))


class BenchmarkCache:
    def __init__(self, table: str, store: SQLCacheStore, key: Tuple[Any, ...]):
        self.table = table
        self.store = store
        self.key = key
        self.memory: Dict[str, Tuple[float, ...]] = {}

    def get(self, config: triton.Config) -> Optional[Tuple[float, ...]]:
        config_key = _serialize_config(config)
        if config_key in self.memory:
            return self.memory[config_key]
        payload = self.store.get_benchmark(self.table, _serialize_tuple_key(self.key), config_key)
        if payload is None:
            return None
        value = tuple(json.loads(payload))
        self.memory[config_key] = value
        return value

    def __setitem__(self, config: triton.Config, value: Tuple[float, ...]):
        config_key = _serialize_config(config)
        self.memory[config_key] = value
        self.store.set_benchmark(
            self.table,
            _serialize_tuple_key(self.key),
            config_key,
            json.dumps(list(value), separators=(",", ":")),
        )


class LibCache(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LibCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        major, minor = _triton_version_parts()
        if db_path is None:
            try:
                device_name = torch_device_fn.get_device_name().replace(" ", "_")
            except AttributeError:
                device_name = device.name
            cache_file_name = f"TunedConfig_{device_name}_triton_{major}_{minor}.db"
            db_path = _config_cache_dir() / cache_file_name
        self.store = SQLCacheStore(db_path)
        self.config_cache_pool: Dict[str, ConfigCache] = {}
        self.benchmark_cache_pool: Dict[Tuple[str, Tuple[Any, ...]], BenchmarkCache] = {}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_config(key)
        return self.get_benchmark(*key)

    def get_config(self, table: str) -> ConfigCache:
        ret = self.config_cache_pool.get(table)
        if ret is None:
            ret = ConfigCache(table, self.store)
            self.config_cache_pool[table] = ret
        return ret

    def get_benchmark(self, table: str, key: Tuple[Any, ...]) -> BenchmarkCache:
        cache_key = (table, key)
        ret = self.benchmark_cache_pool.get(cache_key)
        if ret is None:
            ret = BenchmarkCache(table, self.store, key)
            self.benchmark_cache_pool[cache_key] = ret
        return ret


libcache = LibCache()


class LibTuner(triton.runtime.Autotuner):
    _dispatch_table: Dict[str, Type["LibTuner"]] = {}
    _strategy_table: Dict[str, Callable[[Any], Any]] = {}

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        strategy=None,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
        )
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.__name__ = self.base_fn.__name__
        self.keys = key
        if isinstance(strategy, str):
            strategy = LibTuner.get_strategy(strategy)
        if strategy is None:
            strategy = [LibTuner.get_strategy("default")] * len(self.keys)
        elif not isinstance(strategy, (list, tuple)):
            strategy = [strategy] * len(self.keys)
        self.strategy = [
            LibTuner.get_strategy(s) if isinstance(s, str) else s for s in strategy
        ]
        self.config_table_name = f"{self.__name__}_{self.kernel_hash}"
        self.benchmark_table_name = f"{self.__name__}_{self.cache_key}_benchmark"
        self.cache = libcache[self.config_table_name]

    @cached_property
    def cache_key(self) -> str:
        jit_fn = self.fn
        while not isinstance(jit_fn, triton.runtime.JITFunction):
            jit_fn = jit_fn.fn
        return jit_fn.cache_key

    @cached_property
    def kernel_hash(self) -> str:
        return hashlib.md5(
            f"{self.cache_key}{self.configs_hash}".encode("utf-8")
        ).hexdigest()[:32]

    @cached_property
    def configs_hash(self) -> str:
        return hashlib.md5(
            ",".join(map(str, self.configs)).encode("utf-8")
        ).hexdigest()[:32]

    def get_key(self, args):
        key = tuple(
            self.strategy[idx](args[name]) for idx, name in enumerate(self.keys) if name in args
        )
        key += tuple(str(arg.dtype) for arg in args.values() if hasattr(arg, "dtype"))
        return key

    @staticmethod
    def register(name: str):
        def decorator(subclass):
            LibTuner._dispatch_table[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._dispatch_table[name]

    @classmethod
    def get_strategy(cls, name: str):
        return cls._strategy_table[name]

    @staticmethod
    def register_strategy(name: str):
        def decorator(strategy):
            LibTuner._strategy_table[name] = strategy
            return strategy
        return decorator

    @staticmethod
    def register_policy(name: str):
        def decorator(policy_impl):
            @LibTuner.register(name)
            class AnonymousLibTunerImpl(LibTuner):
                def policy(self, fn, configs, args, kwargs):
                    return policy_impl(fn, configs, args, kwargs)
            return AnonymousLibTunerImpl
        return decorator

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for k, v in all_args.items() if k in self.arg_names}
            key = self.get_key(_args)
            if key not in self.cache:
                used_cached_result = False
                benchmark_cache = libcache[self.benchmark_table_name, key]
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()

                def bench(config: triton.Config) -> List[float]:
                    ret = benchmark_cache.get(config)
                    if ret is None:
                        ret = tuple(self._bench(*args, config=config, **kwargs))
                        benchmark_cache[config] = ret
                    return list(ret)

                best_config, timings = self.policy(bench, pruned_configs, args, kwargs)
                self.bench_time = time.time() - bench_start
                self.cache[key] = best_config
                full_nargs = {**self.nargs, **kwargs, **_config_all_kwargs(best_config)}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
            cached_kwargs = _config_all_kwargs(config)
            for original_config in self.configs:
                if _config_all_kwargs(original_config) == cached_kwargs:
                    config = original_config
                    break
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING") == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **_config_all_kwargs(config)}
            config.pre_hook(full_nargs)
        ret = self.fn.run(*args, **kwargs, **_config_all_kwargs(config))
        self.nargs = None
        return ret


@LibTuner.register_strategy(None)
@LibTuner.register_strategy("default")
def default_strategy(key: Any) -> Any:
    return key


@LibTuner.register_strategy("log")
def log2_strategy(key: Union[int, float]) -> float:
    return 2 ** math.ceil(math.log2(key))


@LibTuner.register_strategy("align32")
def align32_strategy(key: Union[int, float]) -> int:
    return math.ceil(key / 32) * 32


@LibTuner.register_policy("default")
def default_policy(bench_fn, configs: Iterator[triton.Config], args, kwargs):
    timings = {config: bench_fn(config)[0] for config in configs}
    best_config = min(timings, key=timings.get)
    return best_config, timings


def libtuner(
    configs,
    key,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    prune_configs_by=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
    strategy=None,
    policy="default",
):
    reset_to_zero = [] if reset_to_zero is None else reset_to_zero
    restore_value = [] if restore_value is None else restore_value

    def decorator(fn):
        tuner_cls = LibTuner.get(policy)
        return tuner_cls(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            strategy=strategy,
        )

    return decorator
