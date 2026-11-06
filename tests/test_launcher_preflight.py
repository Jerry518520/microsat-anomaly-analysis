from unittest.mock import Mock


def test_occupied_port_is_reported_without_stopping_its_process(monkeypatch):
    from scripts import start_ui
    monkeypatch.setattr(start_ui, "port_in_use", lambda port: True)
    commands = Mock()
    monkeypatch.setattr(start_ui.subprocess, "run", commands)
    assert start_ui.check_port(8000, "API") is False
    commands.assert_not_called()


def test_failed_dependency_check_returns_failure(monkeypatch):
    from scripts import start_ui
    monkeypatch.setattr(start_ui.subprocess, "run", lambda *a, **k: Mock(returncode=1, stderr="missing dependency"))
    assert start_ui.check_api_deps() is False


def test_cleanup_only_stops_its_running_children(monkeypatch):
    from scripts import start_ui
    live, dead = Mock(pid=10), Mock(pid=20)
    live.poll.return_value, dead.poll.return_value = None, 0
    monkeypatch.setattr(start_ui, "child_processes", [live, dead], raising=False)
    stopped = []
    monkeypatch.setattr(start_ui, "kill_pid_tree", stopped.append)
    start_ui.cleanup()
    assert stopped == [10]
