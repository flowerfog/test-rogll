from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automation.autodl_worker import AutoDLWorker
from automation.bootstrap_controller import BootstrapController
from automation.common import (
    AutomationError,
    JobSpec,
    StateStore,
    build_codex_args,
    git,
    load_json,
)
from automation.local_watcher import LocalWatcher


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def toml_string(value: str | Path) -> str:
    return json.dumps(str(value).replace("\\", "/"))


class JobSpecTests(unittest.TestCase):
    def valid_job(self) -> dict[str, object]:
        return {
            "version": 1,
            "job_id": "baseline-v1",
            "enabled": True,
            "command": ["python", "train.py"],
            "timeout_seconds": 60,
            "gpu": "0",
            "artifacts": ["output/*.json"],
            "metrics_file": "output/metrics.json",
            "notes": "test",
            "env": {"PYTHONUNBUFFERED": "1"},
        }

    def parse(self, raw: dict[str, object]) -> JobSpec:
        return JobSpec.from_mapping(
            raw,
            allowed_programs=["python"],
            allowed_env=["PYTHONUNBUFFERED"],
            max_timeout_seconds=3600,
        )

    def test_valid_job(self) -> None:
        job = self.parse(self.valid_job())
        self.assertEqual(job.job_id, "baseline-v1")
        self.assertEqual(job.artifacts, ("output/*.json",))

    def test_rejects_workspace_escape(self) -> None:
        raw = self.valid_job()
        raw["artifacts"] = ["../secret.txt"]
        with self.assertRaises(AutomationError):
            self.parse(raw)

    def test_rejects_unknown_key_and_program(self) -> None:
        raw = self.valid_job()
        raw["typo"] = True
        with self.assertRaises(AutomationError):
            self.parse(raw)
        raw = self.valid_job()
        raw["command"] = ["bash", "-c", "anything"]
        with self.assertRaises(AutomationError):
            self.parse(raw)


class StateStoreTests(unittest.TestCase):
    def test_discovery_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary) / "state.sqlite3")
            try:
                first = store.discover_run("a" * 40)
                store.update_run("a" * 40, "succeeded", job_id="job")
                second = store.discover_run("a" * 40)
                self.assertEqual(first["status"], "discovered")
                self.assertEqual(second["status"], "succeeded")
            finally:
                store.close()


class CodexProviderTests(unittest.TestCase):
    def test_environment_provider_is_injected_without_exposing_key(self) -> None:
        provider_env = {
            "CODEX_PROVIDER_BASE_URL": "https://example.invalid/v1",
            "CODEX_PROVIDER_API_KEY": "secret-value",
            "CODEX_PROVIDER_MODEL": "test-model",
            "CODEX_PROVIDER_ID": "test_provider",
            "CODEX_PROVIDER_REASONING_EFFORT": "high",
        }
        with patch.dict(os.environ, provider_env, clear=False):
            args = build_codex_args(["codex", "exec"])

        self.assertNotIn("--ignore-user-config", args)
        self.assertEqual(args[-2:], ["--model", "test-model"])
        self.assertIn('model_provider="test_provider"', args)
        self.assertIn(
            'model_providers.test_provider.env_key="CODEX_PROVIDER_API_KEY"',
            args,
        )
        self.assertNotIn("secret-value", " ".join(args))

        with patch.dict(os.environ, provider_env, clear=False):
            isolated_args = build_codex_args(
                ["codex", "exec"], ignore_user_config=True
            )
        self.assertIn("--ignore-user-config", isolated_args)

    def test_environment_provider_requires_key(self) -> None:
        provider_env = {
            "CODEX_PROVIDER_BASE_URL": "https://example.invalid/v1",
            "CODEX_PROVIDER_API_KEY": "",
            "CODEX_PROVIDER_MODEL": "test-model",
        }
        with patch.dict(os.environ, provider_env, clear=False):
            with self.assertRaisesRegex(
                AutomationError, "CODEX_PROVIDER_API_KEY is required"
            ):
                build_codex_args(["codex", "exec"])


class WorkerEnvironmentTests(unittest.TestCase):
    def test_worker_static_environment_overrides_job_value(self) -> None:
        worker = AutoDLWorker.__new__(AutoDLWorker)
        worker.inherit_env = ()
        worker.static_env = {
            "ROGII_DATA_ROOT": "/autodl-tmp/rogii-wellbore-geology-prediction"
        }
        job = JobSpec.from_mapping(
            {
                "version": 1,
                "job_id": "env-test",
                "enabled": True,
                "command": ["python", "train.py"],
                "timeout_seconds": 60,
                "gpu": "0",
                "artifacts": [],
                "metrics_file": None,
                "notes": "",
                "env": {"ROGII_DATA_ROOT": "/root/autolab/data"},
            },
            allowed_programs=["python"],
            allowed_env=["ROGII_DATA_ROOT"],
            max_timeout_seconds=300,
        )

        env = worker.execution_env(job)

        self.assertEqual(
            env["ROGII_DATA_ROOT"],
            "/autodl-tmp/rogii-wellbore-geology-prediction",
        )


@unittest.skipUnless(shutil.which("git"), "Git is required")
class BootstrapEndToEndTests(unittest.TestCase):
    def test_empty_remote_gets_first_codex_generated_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "empty-source.git"
            git(["init", "--bare", str(remote)])

            data_root = root / "competition-data"
            train = data_root / "train"
            test = data_root / "test"
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            (train / "well1__horizontal_well.csv").write_text(
                "MD,X,Y,Z,GR,TVT_input\n1,0,0,0,50,10\n2,1,0,0,55,\n",
                encoding="utf-8",
            )
            (train / "well1__typewell.csv").write_text(
                "TVT,GR\n10,50\n11,55\n", encoding="utf-8"
            )
            (test / "well2__horizontal_well.csv").write_text(
                "MD,X,Y,Z,GR,TVT_input\n1,0,0,0,45,8\n2,1,0,0,48,\n",
                encoding="utf-8",
            )
            (test / "well2__typewell.csv").write_text(
                "TVT,GR\n8,45\n9,48\n", encoding="utf-8"
            )
            (data_root / "sample_submission.csv").write_text(
                "id,TVT\nwell2_2,0\n", encoding="utf-8"
            )

            fake_codex = root / "fake_bootstrap_codex.py"
            fake_codex.write_text(
                "import json, pathlib, sys\n"
                "root = pathlib.Path.cwd()\n"
                "files = {\n"
                " 'README.md': '# Baseline\\n',\n"
                " 'requirements.txt': 'numpy\\npandas\\n',\n"
                " 'baseline.py': 'print(\\\"baseline\\\")\\n',\n"
                " 'tests/test_metrics.py': 'def test_placeholder(): assert True\\n',\n"
                " 'autodl/job.json': json.dumps({"
                "'version':1,'job_id':'bootstrap-baseline-v1','enabled':True,"
                "'command':['python','baseline.py'],'timeout_seconds':300,'gpu':'0',"
                "'artifacts':[],'metrics_file':None,'notes':'bootstrap','env':"
                "{'PYTHONUNBUFFERED':'1'}}, indent=2) + '\\n'\n"
                "}\n"
                "for name, content in files.items():\n"
                " p = root / name; p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(content, encoding='utf-8')\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "out.write_text(json.dumps({'decision':'proposed','summary':'first baseline',"
                "'changed_files':sorted(files),'validation':['fake'],"
                "'next_job_id':'bootstrap-baseline-v1'}), encoding='utf-8')\n",
                encoding="utf-8",
            )

            config = root / "bootstrap.toml"
            config.write_text(
                f"""
[bootstrap]
state_dir = {toml_string(root / 'state')}
checkout_dir = {toml_string(root / 'checkout')}
local_data_root = {toml_string(data_root)}
autodl_data_root = "/root/autolab/data"
task_files = [{toml_string(data_root / 'sample_submission.csv')}]
manifest_path = "autodl/job.json"
author_name = "Bootstrap Test"
author_email = "test@example.invalid"

[source]
url = {toml_string(remote)}
branch = "experiments"

[codex]
command = [{toml_string(sys.executable)}, {toml_string(fake_codex)}]
timeout_seconds = 30

[execution]
allowed_programs = ["python"]
allowed_job_env = ["PYTHONUNBUFFERED"]
max_timeout_seconds = 3600
""".strip()
                + "\n",
                encoding="utf-8",
            )

            controller = BootstrapController(config)
            self.assertFalse(controller.ensure_empty_source_checkout())
            self.assertFalse(controller.ensure_empty_source_checkout())
            inventory = controller.build_data_inventory()
            controller.invoke_codex(inventory)
            interrupted_manifest_path = controller.checkout / "autodl" / "job.json"
            interrupted_manifest = {
                "job_id": "bootstrap-baseline-v1",
                "enabled": True,
                "command": "python baseline.py",
            }
            write_json(interrupted_manifest_path, interrupted_manifest)
            interrupted_proposal_path = controller.state_dir / "bootstrap-proposal.json"
            interrupted_proposal = load_json(interrupted_proposal_path)
            interrupted_proposal["decision"] = "blocked"
            write_json(interrupted_proposal_path, interrupted_proposal)
            sha = controller.run()
            readback = root / "readback"
            git(["clone", "--branch", "experiments", str(remote), str(readback)])
            self.assertEqual(git(["rev-parse", "HEAD"], cwd=readback), sha)
            self.assertTrue((readback / "README.md").is_file())
            manifest = load_json(readback / "autodl" / "job.json")
            self.assertEqual(manifest["job_id"], "bootstrap-baseline-v1")
            self.assertFalse(any(readback.rglob("*.csv")))


@unittest.skipUnless(shutil.which("git"), "Git is required")
class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_work = self.root / "source-work"
        self.source_remote = self.root / "source.git"
        self.results_seed = self.root / "results-seed"
        self.results_remote = self.root / "results.git"
        self._create_source_repository()
        self._create_results_repository()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _configure_identity(repository: Path) -> None:
        git(["config", "user.name", "Automation Test"], cwd=repository)
        git(["config", "user.email", "test@example.invalid"], cwd=repository)

    def _create_source_repository(self) -> None:
        git(["init", "--initial-branch", "experiments", str(self.source_work)])
        self._configure_identity(self.source_work)
        script = self.source_work / "train_experiment.py"
        script.write_text(
            "from pathlib import Path\n"
            "import json\n"
            "out = Path('output/test-job')\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'metrics.json').write_text(json.dumps({'pooled_rmse': 6.25}))\n",
            encoding="utf-8",
        )
        write_json(
            self.source_work / "autodl" / "job.json",
            {
                "version": 1,
                "job_id": "test-job",
                "enabled": True,
                "command": [sys.executable, "train_experiment.py"],
                "timeout_seconds": 60,
                "gpu": "0",
                "artifacts": ["output/test-job/metrics.json"],
                "metrics_file": "output/test-job/metrics.json",
                "notes": "end-to-end test",
                "env": {"PYTHONUNBUFFERED": "1"},
            },
        )
        git(["add", "."], cwd=self.source_work)
        git(["commit", "-m", "test experiment"], cwd=self.source_work)
        self.source_sha = git(["rev-parse", "HEAD"], cwd=self.source_work)
        git(["clone", "--bare", str(self.source_work), str(self.source_remote)])

    def _create_results_repository(self) -> None:
        git(["init", "--initial-branch", "main", str(self.results_seed)])
        self._configure_identity(self.results_seed)
        (self.results_seed / "README.md").write_text("# Results\n", encoding="utf-8")
        git(["add", "README.md"], cwd=self.results_seed)
        git(["commit", "-m", "initialize results"], cwd=self.results_seed)
        git(["clone", "--bare", str(self.results_seed), str(self.results_remote)])

    def _autodl_config(self) -> Path:
        config = self.root / "autodl.toml"
        executable_name = Path(sys.executable).name
        config.write_text(
            f"""
[worker]
id = "test-worker"
poll_seconds = 1
state_dir = {toml_string(self.root / 'worker-state')}
work_dir = {toml_string(self.root / 'worker-work')}
manifest_path = "autodl/job.json"

[source]
url = {toml_string(self.source_remote)}
branch = "experiments"
mirror_dir = {toml_string(self.root / 'source-mirror.git')}

[results]
url = {toml_string(self.results_remote)}
branch = "main"
checkout_dir = {toml_string(self.root / 'results-writer')}
author_name = "Test Worker"
author_email = "test@example.invalid"

[execution]
allowed_programs = [{toml_string(executable_name)}]
allowed_job_env = ["PYTHONUNBUFFERED"]
max_timeout_seconds = 300
max_artifact_bytes = 1000000
require_gpu = false
inherit_env = ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME"]

[codex]
enabled = false
command = ["codex", "exec"]
timeout_seconds = 30
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return config

    def _run_worker(self) -> Path:
        config = self._autodl_config()
        dry_worker = AutoDLWorker(config, dry_run=True)
        try:
            self.assertEqual(dry_worker.process_latest(), "dry_run")
        finally:
            dry_worker.close()
        worker = AutoDLWorker(config)
        try:
            self.assertEqual(worker.process_latest(), "succeeded")
            self.assertEqual(worker.process_latest(), "succeeded")
        finally:
            worker.close()
        result_clone = self.root / "result-readback"
        git(["clone", str(self.results_remote), str(result_clone)])
        return result_clone

    def test_worker_executes_and_publishes_once(self) -> None:
        result_clone = self._run_worker()
        run_dir = result_clone / "runs" / "test-job" / self.source_sha[:12]
        record = load_json(run_dir / "run.json")
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["artifact_count"], 1)
        self.assertTrue(
            (run_dir / "artifacts" / "output" / "test-job" / "metrics.json").is_file()
        )
        commit_count = int(git(["rev-list", "--count", "HEAD"], cwd=result_clone))
        self.assertEqual(commit_count, 2)

    def test_codex_rejection_prevents_execution(self) -> None:
        fake_codex = self.root / "fake_reviewer.py"
        fake_codex.write_text(
            "import json, sys\n"
            "out = sys.argv[sys.argv.index('-o') + 1]\n"
            "review = {'decision':'rejected','summary':'blocking test finding',"
            "'risk':'high','findings':[{'severity':'high','blocking':True,"
            "'file':'train_experiment.py','line':1,'message':'test rejection'}]}\n"
            "with open(out, 'w', encoding='utf-8') as f:\n"
            "    json.dump(review, f)\n",
            encoding="utf-8",
        )
        config = self._autodl_config()
        content = config.read_text(encoding="utf-8")
        content = content.replace("enabled = false", "enabled = true")
        content = content.replace(
            'command = ["codex", "exec"]',
            f"command = [{toml_string(sys.executable)}, {toml_string(fake_codex)}]",
        )
        config.write_text(content, encoding="utf-8")
        worker = AutoDLWorker(config)
        try:
            self.assertEqual(worker.process_latest(), "rejected")
        finally:
            worker.close()
        result_clone = self.root / "rejected-readback"
        git(["clone", str(self.results_remote), str(result_clone)])
        run_dir = result_clone / "runs" / "test-job" / self.source_sha[:12]
        record = load_json(run_dir / "run.json")
        self.assertEqual(record["status"], "rejected")
        self.assertIsNone(record["return_code"])
        self.assertEqual(record["artifact_count"], 0)
        self.assertFalse((run_dir / "artifacts").exists())

        retry_worker = AutoDLWorker(config)
        try:
            retry_worker.prepare_retry(self.source_sha)
            self.assertEqual(retry_worker.process_latest(), "rejected")
        finally:
            retry_worker.close()
        retry_readback = self.root / "retry-readback"
        git(["clone", str(self.results_remote), str(retry_readback)])
        retry_record = load_json(run_dir := retry_readback / "runs" / "test-job" / self.source_sha[:12] / "run.json")
        self.assertEqual(retry_record["status"], "rejected")
        self.assertEqual(int(git(["rev-list", "--count", "HEAD"], cwd=retry_readback)), 3)

    def test_worker_reports_nonzero_exit(self) -> None:
        (self.source_work / "train_experiment.py").write_text(
            "raise SystemExit(7)\n", encoding="utf-8"
        )
        manifest_path = self.source_work / "autodl" / "job.json"
        manifest = load_json(manifest_path)
        manifest["job_id"] = "failing-job"
        manifest["artifacts"] = []
        manifest["metrics_file"] = None
        write_json(manifest_path, manifest)
        git(["add", "."], cwd=self.source_work)
        git(["commit", "-m", "failing experiment"], cwd=self.source_work)
        git(["remote", "add", "origin", str(self.source_remote)], cwd=self.source_work)
        git(["push", "origin", "experiments"], cwd=self.source_work)
        failing_sha = git(["rev-parse", "HEAD"], cwd=self.source_work)

        worker = AutoDLWorker(self._autodl_config())
        try:
            self.assertEqual(worker.process_latest(), "failed")
        finally:
            worker.close()
        result_clone = self.root / "failed-readback"
        git(["clone", str(self.results_remote), str(result_clone)])
        record = load_json(
            result_clone / "runs" / "failing-job" / failing_sha[:12] / "run.json"
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["return_code"], 7)
        self.assertEqual(record["message"], "job exited with 7")

    def test_local_watcher_analyzes_new_result(self) -> None:
        self._run_worker()
        fake_codex = self.root / "fake_codex.py"
        fake_codex.write_text(
            "import json, sys\n"
            "out = sys.argv[sys.argv.index('-o') + 1]\n"
            "with open(out, 'w', encoding='utf-8') as f:\n"
            "    json.dump({'decision':'no_change','summary':'done','changed_files':[],"
            "'validation':[],'next_job_id':None}, f)\n",
            encoding="utf-8",
        )
        config = self.root / "local.toml"
        config.write_text(
            f"""
[watcher]
poll_seconds = 1
state_dir = {toml_string(self.root / 'local-state')}
ignore_existing_on_first_run = false

[source]
url = {toml_string(self.source_remote)}
branch = "experiments"
checkout_dir = {toml_string(self.root / 'source-controller')}

[results]
url = {toml_string(self.results_remote)}
branch = "main"
checkout_dir = {toml_string(self.root / 'results-reader')}

[codex]
command = [{toml_string(sys.executable)}, {toml_string(fake_codex)}]
timeout_seconds = 30

[proposal]
mode = "analysis"
max_rounds = 2
result_statuses = ["succeeded"]
manifest_path = "autodl/job.json"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        watcher = LocalWatcher(config)
        try:
            self.assertTrue(watcher.poll_once())
            event = watcher.store.get_event(self.source_sha)
            self.assertIsNotNone(event)
            self.assertEqual(event["status"], "no_change")
            self.assertTrue(watcher.poll_once())
        finally:
            watcher.close()


if __name__ == "__main__":
    unittest.main()
