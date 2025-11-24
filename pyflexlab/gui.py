from __future__ import annotations

import ast
import inspect
import logging
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, get_args, get_origin

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    from PyQt6 import QtWebEngineWidgets
except ImportError:  # pragma: no cover - optional dependency
    QtWebEngineWidgets = None

from pyflexlab.measure_flow import MeasureFlow


WIN11_ACCENT = "#2563eb"
WIN11_SURFACE = "#1f2933"
WIN11_SURFACE_DARK = "#151c24"
WIN11_TEXT = "#e5e7eb"
SIGNATURE_EMPTY = inspect.Signature.empty
PARAM_INPUTS_PER_ROW = 5


@dataclass(frozen=True)
class InstrumentConfig:
    id: str
    label: str
    method: str
    mode: str
    meter_code: Optional[str] = None
    min_addresses: int = 1
    placeholder: str = ""
    channel_required: bool = False
    channel_label: str = "Channel"
    channel_default: int = 1


INSTRUMENT_CONFIGS: Tuple[InstrumentConfig, ...] = (
    InstrumentConfig(
        id="meter_6221",
        label="Keithley 6221",
        method="load_meter",
        mode="addresses",
        meter_code="6221",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_2182",
        label="Keithley 2182",
        method="load_meter",
        mode="addresses",
        meter_code="2182",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_2182a",
        label="Keithley 2182A",
        method="load_meter",
        mode="addresses",
        meter_code="2182a",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_2400",
        label="Keithley 2400",
        method="load_meter",
        mode="addresses",
        meter_code="2400",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_2401",
        label="Keithley 2401",
        method="load_meter",
        mode="addresses",
        meter_code="2401",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_2450",
        label="Keithley 2450",
        method="load_meter",
        mode="addresses",
        meter_code="2450",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_6430",
        label="Keithley 6430",
        method="load_meter",
        mode="addresses",
        meter_code="6430",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_sr830",
        label="SR830 Lock-in",
        method="load_meter",
        mode="addresses",
        meter_code="sr830",
        min_addresses=2,
        placeholder="Two GPIB addresses separated by comma",
    ),
    InstrumentConfig(
        id="meter_sr860",
        label="SR860 Lock-in",
        method="load_meter",
        mode="addresses",
        meter_code="sr860",
        placeholder="Comma-separated GPIB addresses",
    ),
    InstrumentConfig(
        id="meter_b2902",
        label="Keysight B2902",
        method="load_meter",
        mode="addresses",
        meter_code="b2902",
        placeholder="Comma-separated addresses (instrument treats as channel pair)",
        channel_required=True,
        channel_label="Channel",
        channel_default=1,
    ),
    InstrumentConfig(
        id="meter_b2902b",
        label="Keysight B2902B",
        method="load_meter",
        mode="addresses",
        meter_code="b2902b",
        placeholder="Comma-separated addresses (instrument treats as channel pair)",
        channel_required=True,
        channel_label="Channel",
        channel_default=1,
    ),
    InstrumentConfig(
        id="meter_b2902ch",
        label="Keysight B2902 (Channel Mode)",
        method="load_meter",
        mode="addresses",
        meter_code="b2902ch",
        placeholder="Comma-separated addresses",
        channel_required=True,
        channel_label="Channel",
        channel_default=1,
    ),
    InstrumentConfig(
        id="meter_tests",
        label="Simulated Test Meters",
        method="load_meter",
        mode="addresses",
        meter_code="tests",
        placeholder="Leave blank or provide mock addresses",
    ),
    InstrumentConfig(
        id="mercury_ips",
        label="Mercury iPS",
        method="load_mercury_ips",
        mode="address_single",
        placeholder="TCPIP address (e.g. TCPIP0::...::SOCKET)",
    ),
    InstrumentConfig(
        id="mercury_itc",
        label="Mercury ITC",
        method="load_mercury_itc",
        mode="address_single",
        placeholder="TCPIP address",
    ),
    InstrumentConfig(
        id="itc503",
        label="ITC503 (Upper/Lower)",
        method="load_ITC503",
        mode="address_pair",
        placeholder="Upper,Lower GPIB addresses",
    ),
    InstrumentConfig(
        id="lakeshore",
        label="Lakeshore ITC",
        method="load_lakeshore",
        mode="address_single",
        placeholder="GPIB address (e.g. GPIB0::12::INSTR)",
    ),
    InstrumentConfig(
        id="laser",
        label="Laser System",
        method="load_laser",
        mode="trigger",
    ),
    InstrumentConfig(
        id="rotator",
        label="Rotator Probe",
        method="load_rotator",
        mode="trigger",
    ),
    InstrumentConfig(
        id="fakes",
        label="Load Fake Instruments",
        method="load_fakes",
        mode="count",
        placeholder="Number of fake meters",
    ),
)


def parse_addresses(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


class DashBridge:
    """Lightweight wrapper that references an external Dash server."""

    def __init__(self, port: int = 11235) -> None:
        self.port = port

    def start(self) -> None:
        """No-op retained for compatibility."""

    def append_log(self, message: str) -> None:
        """External Dash server manages its own logs; nothing to forward."""


class LogEmitter(QtCore.QObject):
    """Qt signal emitter for log messages."""

    log_signal = QtCore.pyqtSignal(str)


class ExceptionEmitter(QtCore.QObject):
    """Qt signal emitter for unhandled exceptions."""

    exception_signal = QtCore.pyqtSignal(str, str, str)


class GuiLogHandler(logging.Handler):
    """Forward log records to the Qt GUI and Dash board."""

    def __init__(self, emitter: LogEmitter, dash_bridge: DashBridge) -> None:
        super().__init__()
        self._emitter = emitter
        self._dash_bridge = dash_bridge

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self._emitter.log_signal.emit(message)
        self._dash_bridge.append_log(message)


class ParameterForm(QtWidgets.QWidget):
    """Dynamic form that reflects the signature of a selected MeasureFlow method."""

    def __init__(self) -> None:
        super().__init__()
        self._grid_layout = QtWidgets.QGridLayout()
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(12)
        self._grid_layout.setVerticalSpacing(12)
        self.setLayout(self._grid_layout)
        self._fields: Dict[str, QtWidgets.QWidget] = {}
        self._parameters: Dict[str, inspect.Parameter] = {}
        self._extractors: Dict[str, Callable[[], Tuple[Any, bool, bool]]] = {}
        self._instrument_provider: Optional[Callable[[], List[Tuple[str, Any]]]] = None
        self._instrument_fields: Dict[str, InstrumentSelector] = {}
        self._literal_fields: Dict[str, QtWidgets.QComboBox] = {}

    @property
    def parameters(self) -> Dict[str, inspect.Parameter]:
        return self._parameters

    @property
    def fields(self) -> Dict[str, QtWidgets.QWidget]:
        return self._fields

    def set_instrument_provider(
        self, provider: Optional[Callable[[], List[Tuple[str, Any]]]]
    ) -> None:
        self._instrument_provider = provider
        self.refresh_instrument_fields()

    def refresh_instrument_fields(self) -> None:
        if not self._instrument_provider:
            return
        options = self._instrument_provider()
        for selector in self._instrument_fields.values():
            selector.set_options(options)

    def extract_value(self, name: str) -> Tuple[Any, bool, bool]:
        extractor = self._extractors.get(name)
        if extractor is None:
            return "", True, True
        return extractor()

    def set_method(self, method: Callable[..., Any]) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._fields.clear()
        self._parameters.clear()
        self._extractors.clear()
        self._instrument_fields.clear()
        self._literal_fields.clear()

        signature = inspect.signature(method)
        position = 0
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue

            placeholder = self._build_placeholder(parameter)

            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)

            label = QtWidgets.QLabel(name)
            label.setStyleSheet("font-size: 12px; color: #9ca3af;")

            container_layout.addWidget(label)

            literal_options = self._literal_options(parameter)

            if literal_options is not None:
                combo = QtWidgets.QComboBox()
                combo.addItems([str(option) for option in literal_options])
                container_layout.addWidget(combo)
                widget = combo
                self._literal_fields[name] = combo

                def make_literal_extractor(
                    combo_box: QtWidgets.QComboBox, options: Tuple[Any, ...]
                ) -> Callable[[], Tuple[Any, bool, bool]]:
                    def extractor() -> Tuple[Any, bool, bool]:
                        idx = combo_box.currentIndex()
                        if idx < 0 or idx >= len(options):
                            return None, True, False
                        return options[idx], False, False

                    return extractor

                self._extractors[name] = make_literal_extractor(combo, literal_options)
            elif self._is_instrument_parameter(name, parameter):
                allow_multiple, max_count = self._instrument_selection_rules(
                    name, parameter
                )
                selector = InstrumentSelector(
                    allow_multiple=allow_multiple, max_count=max_count
                )
                if self._instrument_provider:
                    selector.set_options(self._instrument_provider())
                container_layout.addWidget(selector)
                widget = selector
                self._instrument_fields[name] = selector
                self._extractors[name] = selector.extract_value
            else:
                field_input = QtWidgets.QLineEdit()
                field_input.setPlaceholderText(placeholder)
                field_input.setClearButtonEnabled(True)
                field_input.setMaximumWidth(220)
                container_layout.addWidget(field_input)
                widget = field_input

                def make_extractor(
                    line_edit: QtWidgets.QLineEdit,
                ) -> Callable[[], Tuple[Any, bool, bool]]:
                    def extractor() -> Tuple[Any, bool, bool]:
                        raw_value = line_edit.text().strip()
                        return raw_value, raw_value == "", True

                    return extractor

                self._extractors[name] = make_extractor(field_input)

            container.setLayout(container_layout)

            row = position // PARAM_INPUTS_PER_ROW
            column = position % PARAM_INPUTS_PER_ROW
            self._grid_layout.addWidget(container, row, column)
            position += 1

            self._fields[name] = widget
            self._parameters[name] = parameter

    @staticmethod
    def _build_placeholder(parameter: inspect.Parameter) -> str:
        annotation = parameter.annotation
        readable_annotation = ""
        if annotation is not SIGNATURE_EMPTY:
            if isinstance(annotation, str):
                readable_annotation = annotation
            else:
                readable_annotation = getattr(annotation, "__name__", str(annotation))

        default = parameter.default
        if default is SIGNATURE_EMPTY:
            default_repr = "required"
        else:
            default_repr = f"default: {default!r}"

        if readable_annotation:
            return f"{readable_annotation} ({default_repr})"
        return default_repr

    @staticmethod
    def _is_instrument_parameter(name: str, parameter: inspect.Parameter) -> bool:
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ("meter", "instr")):
            return True
        if parameter.annotation is SIGNATURE_EMPTY:
            return False
        annotation_str = repr(parameter.annotation)
        return "Meter" in annotation_str or "SourceMeter" in annotation_str

    @staticmethod
    def _instrument_selection_rules(
        name: str, parameter: inspect.Parameter
    ) -> tuple[bool, Optional[int]]:
        if parameter.annotation is SIGNATURE_EMPTY:
            return False, None
        annotation = parameter.annotation
        max_count: Optional[int] = None
        allow_multiple = False
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin in (list, List, Sequence, tuple, Tuple):
            allow_multiple = True
            if origin is tuple and args:
                max_count = len(args)
            elif args:
                literal_lengths = [
                    len(getattr(arg, "__args__", ()))
                    for arg in args
                    if getattr(arg, "__origin__", None) is tuple
                ]
                if literal_lengths:
                    max_count = max(literal_lengths)
        elif isinstance(annotation, str):
            lower_repr = annotation.lower()
            allow_multiple = any(token in lower_repr for token in ("list", "sequence"))
        else:
            lower_repr = repr(annotation).lower()
            allow_multiple = any(token in lower_repr for token in ("list", "sequence"))
        if not allow_multiple and "list" in name.lower():
            allow_multiple = True
        return allow_multiple, max_count

    @staticmethod
    def _literal_options(parameter: inspect.Parameter) -> Optional[Tuple[Any, ...]]:
        annotation = parameter.annotation
        if annotation is SIGNATURE_EMPTY:
            return None
        origin = get_origin(annotation)
        if origin is None:
            return None
        if getattr(origin, "__name__", "") != "Literal":
            return None
        return get_args(annotation)


class InstrumentSelector(QtWidgets.QWidget):
    """Widget providing dropdown or multi-select for instrument choices."""

    def __init__(self, *, allow_multiple: bool, max_count: Optional[int]) -> None:
        super().__init__()
        self._allow_multiple = allow_multiple
        self._max_count = max_count
        self._options: List[Tuple[str, Any]] = []
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if allow_multiple:
            self._list = QtWidgets.QListWidget()
            self._list.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
            )
            if max_count is not None:
                self._selection_hint = QtWidgets.QLabel(
                    f"Select up to {max_count} instruments."
                )
                self._selection_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
                layout.addWidget(self._selection_hint)
            layout.addWidget(self._list)
        else:
            self._combo = QtWidgets.QComboBox()
            layout.addWidget(self._combo)
        self.setLayout(layout)
        self.set_options([])

    def set_options(self, options: List[Tuple[str, Any]]) -> None:
        self._options = [(label, obj) for label, obj in options if obj is not None]
        if self._allow_multiple:
            selected_objects = {
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in self._list.selectedItems()
            }
            self._list.clear()
            for label, obj in self._options:
                list_item = QtWidgets.QListWidgetItem(label)
                list_item.setData(QtCore.Qt.ItemDataRole.UserRole, obj)
                self._list.addItem(list_item)
                if obj in selected_objects:
                    list_item.setSelected(True)
        else:
            combo = self._combo
            current_obj = combo.currentData() if combo.count() else None
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select instrument...", None)
            selected_index = 0
            for idx, (label, obj) in enumerate(self._options, start=1):
                combo.addItem(label, obj)
                if obj is current_obj:
                    selected_index = idx
            combo.setCurrentIndex(selected_index)
            combo.blockSignals(False)

    def extract_value(self) -> Tuple[Any, bool, bool]:
        if self._allow_multiple:
            selected_items = [
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in self._list.selectedItems()
            ]
            if self._max_count is not None and len(selected_items) > self._max_count:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Too Many Instruments",
                    f"Please select at most {self._max_count} instruments.",
                )
                selected_items = selected_items[: self._max_count]
            return selected_items, len(selected_items) == 0, False
        data = self._combo.currentData()
        return data, data is None, False


class InstrumentDialog(QtWidgets.QDialog):
    """Modal dialog for configuring and initializing instruments."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        measure_flow_getter: Callable[[], Optional[MeasureFlow]],
        status_callback: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Instrument Initialization")
        self.resize(720, 520)
        self._measure_flow_getter = measure_flow_getter
        self._status_callback = status_callback
        self._instrument_controls: Dict[str, Dict[str, Any]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "Select the instruments to initialize and provide the required addresses. "
            "Leave an address blank to skip that instrument."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #9ca3af;")
        layout.addWidget(info_label)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        container.setLayout(container_layout)

        for config in INSTRUMENT_CONFIGS:
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            checkbox = QtWidgets.QCheckBox(config.label)
            row_layout.addWidget(checkbox)

            control: Dict[str, Any] = {"checkbox": checkbox, "config": config}

            if config.mode in {"addresses", "address_single", "address_pair"}:
                line_edit = QtWidgets.QLineEdit()
                line_edit.setPlaceholderText(config.placeholder or "")
                line_edit.setToolTip(config.placeholder or "")
                line_edit.setEnabled(False)
                checkbox.toggled.connect(line_edit.setEnabled)
                row_layout.addWidget(line_edit, 1)
                control["input"] = line_edit
                if config.channel_required:
                    channel_spin = QtWidgets.QSpinBox()
                    channel_spin.setRange(1, 4)
                    channel_spin.setValue(config.channel_default)
                    channel_spin.setEnabled(False)
                    checkbox.toggled.connect(channel_spin.setEnabled)
                    channel_label = QtWidgets.QLabel(config.channel_label)
                    channel_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
                    channel_label.setEnabled(False)

                    def _toggle_label(enabled: bool, label: QtWidgets.QLabel = channel_label) -> None:
                        label.setEnabled(enabled)

                    checkbox.toggled.connect(_toggle_label)
                    row_layout.addWidget(channel_label)
                    row_layout.addWidget(channel_spin)
                    control["channel"] = channel_spin
            elif config.mode == "count":
                spin_box = QtWidgets.QSpinBox()
                spin_box.setMinimum(1)
                spin_box.setMaximum(1_000_000)
                spin_box.setValue(1)
                spin_box.setEnabled(False)
                checkbox.toggled.connect(spin_box.setEnabled)
                row_layout.addWidget(spin_box)
                control["spin"] = spin_box
            else:
                row_layout.addStretch()

            row_layout.addStretch()
            row_widget.setLayout(row_layout)
            container_layout.addWidget(row_widget)
            self._instrument_controls[config.id] = control

        container_layout.addStretch()
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        init_button = QtWidgets.QPushButton("Initialize Selected")
        init_button.setObjectName("accent-button")
        init_button.clicked.connect(self._initialize_instruments)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(init_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def _initialize_instruments(self) -> None:
        try:
            measure_flow = self._measure_flow_getter()
            if measure_flow is None:
                return

            selected_controls = [
                control
                for control in self._instrument_controls.values()
                if control["checkbox"].isChecked()
            ]
            if not selected_controls:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Instruments Selected",
                    "Select at least one instrument to initialize.",
                )
                return

            successes: List[str] = []
            attempted = False
            for control in selected_controls:
                config: InstrumentConfig = control["config"]
                try:
                    if config.mode == "addresses":
                        addresses = parse_addresses(control["input"].text())
                        if not addresses:
                            continue
                        attempted = True
                        if len(addresses) < config.min_addresses:
                            raise ValueError(
                                f"Provide at least {config.min_addresses} address(es)."
                            )
                        call_kwargs: Dict[str, Any] = {}
                        if config.channel_required and "channel" in control:
                            call_kwargs["channel"] = control["channel"].value()
                        getattr(measure_flow, config.method)(
                            config.meter_code, *addresses, **call_kwargs
                        )
                    elif config.mode == "address_single":
                        addresses = parse_addresses(control["input"].text())
                        if not addresses:
                            continue
                        attempted = True
                        if len(addresses) != 1:
                            raise ValueError("Provide exactly one address.")
                        getattr(measure_flow, config.method)(address=addresses[0])
                    elif config.mode == "address_pair":
                        addresses = parse_addresses(control["input"].text())
                        if not addresses:
                            continue
                        attempted = True
                        if len(addresses) != 2:
                            raise ValueError("Provide two addresses separated by a comma.")
                        getattr(measure_flow, config.method)(addresses[0], addresses[1])
                    elif config.mode == "count":
                        count = control["spin"].value()
                        attempted = True if count else attempted
                        getattr(measure_flow, config.method)(no_meters=count)
                    else:
                        attempted = True
                        getattr(measure_flow, config.method)()
                except Exception as exc:  # pylint: disable=broad-except
                    logging.getLogger(__name__).exception(
                        "Failed to initialize %s", config.label
                    )
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Initialization Error",
                        f"Failed to initialize {config.label}: {exc}",
                    )
                    self._status_callback(f"{config.label} initialization failed.")
                else:
                    successes.append(config.label)

            if successes:
                summary = f"Initialized: {', '.join(successes)}"
                self._status_callback(summary)
                QtWidgets.QMessageBox.information(
                    self, "Initialization Complete", summary
                )
            elif not attempted:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Addresses Provided",
                    "No instruments were initialized. Provide addresses or inputs for the selected items.",
                )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover  # pylint: disable=broad-except
            logging.getLogger(__name__).exception(
                "Unexpected error during instrument initialization."
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Initialization Error",
                f"An unexpected error occurred during instrument initialization:\n{exc}",
            )

class MeasureRunner(threading.Thread):
    """Execute a measurement in a background thread."""

    def __init__(
        self,
        method_name: str,
        method: Callable[..., Any],
        kwargs: Dict[str, Any],
        on_finished: Callable[[str], None],
        on_error: Callable[[str, Exception, str], None],
        error_log_path: Path,
    ) -> None:
        super().__init__(daemon=True)
        self._method_name = method_name
        self._method = method
        self._kwargs = kwargs
        self._on_finished = on_finished
        self._on_error = on_error
        self._error_log_path = error_log_path

    def run(self) -> None:
        try:
            self._method(**self._kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            error_traceback = traceback.format_exc()
            log_file = self._save_measurement_error_log(exc, error_traceback)
            self._on_error(self._method_name, exc, log_file)
        else:
            self._on_finished(self._method_name)

    def _save_measurement_error_log(
        self, exc: Exception, error_traceback: str
    ) -> str:
        """Save measurement error log to a file and return the file path."""
        try:
            self._error_log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self._error_log_path / f"measurement_error_{timestamp}.log"

            with open(log_file, "w", encoding="utf-8") as f:
                f.write("PyFlexLab Measurement Error Log\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Method: {self._method_name}\n")
                f.write(f"Error: {type(exc).__name__}: {exc}\n")
                f.write(f"Parameters: {self._kwargs}\n")
                f.write("=" * 80 + "\n\n")
                f.write("Traceback:\n")
                f.write(error_traceback)
                f.write("\n" + "=" * 80 + "\n")

            return str(log_file)
        except Exception as save_exc:  # noqa: BLE001  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("Failed to save measurement error log.")
            return f"Failed to save error log: {save_exc}"


class MeasureFlowGui(QtWidgets.QMainWindow):
    """Main window orchestrating MeasureFlow operations via a PyQt6 interface."""

    def __init__(self, dash_bridge: DashBridge) -> None:
        super().__init__()
        self.setWindowTitle("PyFlexLab Measurement Studio")
        self.resize(1400, 900)

        self._dash_bridge = dash_bridge
        self._dash_bridge.start()

        self._log_emitter = LogEmitter()
        self._log_emitter.log_signal.connect(self._append_log)

        self._exception_emitter = ExceptionEmitter()
        self._exception_emitter.exception_signal.connect(self._handle_exception_signal)

        self._log_handler = GuiLogHandler(self._log_emitter, self._dash_bridge)
        self._log_handler.setLevel(logging.WARNING)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._log_handler)

        self._measure_methods = self._collect_methods()
        self._measure_methods_map = dict(self._measure_methods)
        self._measure_flow: Optional[MeasureFlow] = None
        self._active_project_name: Optional[str] = None
        self._active_out_db_path: Optional[str] = None
        self._active_thread: Optional[MeasureRunner] = None
        self._error_log_path = Path.home() / "pyflexlab_error_logs"

        self._build_ui()
        self._apply_dark_palette()
        self._install_exception_handler()

    def _collect_methods(self) -> List[Tuple[str, Callable[..., Any]]]:
        members = inspect.getmembers(MeasureFlow, predicate=inspect.isfunction)
        return sorted(
            [
                (name, member)
                for name, member in members
                if name.startswith("measure_")
            ],
            key=lambda item: item[0],
        )

    def _install_exception_handler(self) -> None:
        """Install a global exception handler to catch unhandled exceptions."""
        sys.excepthook = self._global_exception_hook

    def _global_exception_hook(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
    ) -> None:
        """Global exception handler that catches all unhandled exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        error_title = f"{exc_type.__name__}: {exc_value}"

        log_file = self._save_error_log(error_msg, error_title)

        self._exception_emitter.exception_signal.emit(error_title, error_msg, log_file)

    def _save_error_log(self, error_msg: str, error_title: str) -> str:
        """Save error log to a file and return the file path."""
        try:
            self._error_log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self._error_log_path / f"error_{timestamp}.log"

            with open(log_file, "w", encoding="utf-8") as f:
                f.write("PyFlexLab Error Log\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {error_title}\n")
                f.write("=" * 80 + "\n\n")
                f.write(error_msg)
                f.write("\n" + "=" * 80 + "\n")

            return str(log_file)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("Failed to save error log.")
            return f"Failed to save error log: {exc}"

    def _handle_exception_signal(
        self, error_title: str, error_msg: str, log_file: str
    ) -> None:
        """Handle exception signal in the Qt main thread."""
        logging.getLogger(__name__).error("Unhandled exception: %s", error_title)
        logging.getLogger(__name__).error("Error details:\n%s", error_msg)

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Application Error")
        msg_box.setText(f"An unexpected error occurred:\n\n{error_title}")
        msg_box.setInformativeText(
            f"The error has been logged to:\n{log_file}\n\n"
            "The application will attempt to continue running. "
            "If you experience further issues, please restart the application."
        )
        msg_box.setDetailedText(error_msg)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Close
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)

        result = msg_box.exec()

        if result == QtWidgets.QMessageBox.StandardButton.Close:
            self._reset_to_initial_state()

        self._append_log(f"ERROR: {error_title}")
        self._append_log(f"Error log saved to: {log_file}")

    def _reset_to_initial_state(self) -> None:
        """Reset the GUI to its initial state after an error."""
        try:
            self._status_label.setText("Ready (reset after error).")
            self._run_button.setEnabled(True)
            self._active_thread = None

            self._method_list.clearSelection()

            self._append_log("GUI reset to initial state after error.")
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("Failed to reset GUI to initial state.")
            self._append_log(f"Failed to reset GUI: {exc}")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout()
        central.setLayout(main_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self._method_list = QtWidgets.QListWidget()
        self._method_list.setMinimumWidth(280)
        self._method_list.itemSelectionChanged.connect(self._handle_method_selection)

        for name, function in self._measure_methods:
            item = QtWidgets.QListWidgetItem(name)
            doc = inspect.getdoc(function) or "No description available."
            item.setToolTip(doc)
            self._method_list.addItem(item)

        splitter.addWidget(self._method_list)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        top_panel = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)

        header_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("Measurement Parameters")
        title_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self._instrument_button = QtWidgets.QPushButton("Configure Instruments")
        self._instrument_button.clicked.connect(self._open_instrument_dialog)
        header_layout.addWidget(self._instrument_button)

        self._cleanup_button = QtWidgets.QPushButton("Force Cleanup")
        self._cleanup_button.clicked.connect(self._manual_cleanup)
        self._cleanup_button.setToolTip(
            "Manually clean up measurement state. "
            "Use this if measurements seem stuck or won't restart properly."
        )
        header_layout.addWidget(self._cleanup_button)

        self._open_dash_button = QtWidgets.QPushButton("Open Dash Panel")
        self._open_dash_button.clicked.connect(self._open_dash_in_browser)
        header_layout.addWidget(self._open_dash_button)

        top_layout.addLayout(header_layout)

        config_widget = QtWidgets.QWidget()
        config_layout = QtWidgets.QFormLayout()
        config_widget.setLayout(config_layout)

        self._project_name_input = QtWidgets.QLineEdit()
        self._project_name_input.setPlaceholderText("Project name (required)")
        config_layout.addRow("Project Name", self._project_name_input)

        custom_path_layout = QtWidgets.QHBoxLayout()
        self._custom_db_input = QtWidgets.QLineEdit()
        self._custom_db_input.setPlaceholderText("Select custom out_db_path (required)")
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_custom_path)
        custom_path_layout.addWidget(self._custom_db_input)
        custom_path_layout.addWidget(browse_button)

        config_layout.addRow("Out DB Path", custom_path_layout)

        top_layout.addWidget(config_widget)

        self._form_container = ParameterForm()
        self._form_container.set_instrument_provider(self._get_instrument_options)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._form_container)
        scroll_area.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        top_layout.addWidget(scroll_area, stretch=1)

        controls_layout = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel("Ready.")
        self._status_label.setStyleSheet("color: #9ca3af;")

        self._run_button = QtWidgets.QPushButton("Run Measurement")
        self._run_button.setObjectName("accent-button")
        self._run_button.clicked.connect(self._run_measurement)

        controls_layout.addWidget(self._status_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self._run_button)

        top_layout.addLayout(controls_layout)

        top_panel.setLayout(top_layout)
        right_splitter.addWidget(top_panel)

        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        log_container = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        self._log_output = QtWidgets.QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumBlockCount(2000)
        self._log_output.setMinimumHeight(180)
        self._log_output.setPlaceholderText("Warnings and errors will appear here...")
        log_layout.addWidget(self._log_output)
        log_container.setLayout(log_layout)
        bottom_splitter.addWidget(log_container)

        if QtWebEngineWidgets is not None:
            dash_container = QtWidgets.QWidget()
            dash_layout = QtWidgets.QVBoxLayout()
            dash_layout.setContentsMargins(0, 0, 0, 0)
            dash_layout.setSpacing(6)
            self._dash_view = QtWebEngineWidgets.QWebEngineView()
            self._dash_view.setUrl(
                QtCore.QUrl(f"http://127.0.0.1:{self._dash_bridge.port}")
            )
            self._dash_view.setMinimumHeight(260)
            dash_layout.addWidget(self._dash_view)
            dash_container.setLayout(dash_layout)
            bottom_splitter.addWidget(dash_container)
        else:
            info_label = QtWidgets.QLabel(
                "Qt WebEngine is not available. Use the 'Open Dash Panel' button to view Dash output in your browser."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #9ca3af; font-style: italic;")
            placeholder_container = QtWidgets.QWidget()
            placeholder_layout = QtWidgets.QVBoxLayout()
            placeholder_layout.setContentsMargins(0, 0, 0, 0)
            placeholder_layout.addWidget(info_label)
            placeholder_container.setLayout(placeholder_layout)
            bottom_splitter.addWidget(placeholder_container)

        right_splitter.addWidget(bottom_splitter)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 2)

    def _open_instrument_dialog(self) -> None:
        dialog = InstrumentDialog(
            self,
            measure_flow_getter=self._prepare_measure_flow,
            status_callback=self._status_label.setText,
        )
        try:
            dialog.exec()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover  # pylint: disable=broad-except
            logging.getLogger(__name__).exception(
                "Unexpected error while displaying instrument dialog."
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Instrument Dialog Error",
                f"An unexpected error occurred while opening the instrument dialog:\n{exc}",
            )
        self._form_container.refresh_instrument_fields()

    def _prepare_measure_flow(self) -> Optional[MeasureFlow]:
        project_name = self._project_name_input.text().strip()
        if not project_name:
            QtWidgets.QMessageBox.critical(
                self, "Missing Project Name", "Provide a project name."
            )
            return None

        custom_db_path = self._custom_db_input.text().strip()
        if not custom_db_path:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing out_db_path",
                (
                    "Please specify the out_db_path via the provided field. "
                    "The GUI intentionally does not fall back to environment defaults."
                ),
            )
            return None

        custom_db_path_resolved = str(
            Path(custom_db_path).expanduser().resolve(strict=False)
        )
        try:
            Path(custom_db_path_resolved).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Path",
                f"Unable to prepare the out_db_path directory:\n{exc}",
            )
            return None

        self._custom_db_input.setText(custom_db_path_resolved)

        if (
            self._measure_flow is None
            or self._active_project_name != project_name
            or self._active_out_db_path != custom_db_path_resolved
        ):
            try:
                self._measure_flow = MeasureFlow(
                    project_name, custom_db_path=custom_db_path_resolved
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.getLogger(__name__).exception("Failed to initialize MeasureFlow.")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Initialization Error",
                    f"Failed to initialize MeasureFlow: {exc}",
                )
                return None
            self._active_project_name = project_name
            self._active_out_db_path = custom_db_path_resolved

        self._form_container.refresh_instrument_fields()
        return self._measure_flow

    def _get_instrument_options(self) -> List[Tuple[str, Any]]:
        if self._measure_flow is None:
            return []
        options: List[Tuple[str, Any]] = []
        for name, entry in self._measure_flow.instrs.items():
            if isinstance(entry, list):
                for idx, inst in enumerate(entry):
                    options.append((f"{name}[{idx}] ({type(inst).__name__})", inst))
            elif entry is not None:
                options.append((f"{name} ({type(entry).__name__})", entry))
        return options

    def _apply_dark_palette(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.setStyle("Fusion")

        palette = QtGui.QPalette()
        palette.setColor(
            QtGui.QPalette.ColorRole.Window, QtGui.QColor(WIN11_SURFACE_DARK)
        )
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(WIN11_TEXT))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#0f172a"))
        palette.setColor(
            QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(WIN11_SURFACE)
        )
        palette.setColor(
            QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(WIN11_SURFACE)
        )
        palette.setColor(
            QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(WIN11_TEXT)
        )
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(WIN11_TEXT))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(WIN11_SURFACE))
        palette.setColor(
            QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(WIN11_TEXT)
        )
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(WIN11_ACCENT))
        palette.setColor(
            QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff")
        )

        app.setPalette(palette)

        stylesheet = f"""
            QWidget {{
                background-color: {WIN11_SURFACE_DARK};
                color: {WIN11_TEXT};
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: {WIN11_SURFACE};
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 6px 10px;
                selection-background-color: {WIN11_ACCENT};
            }}
            QLineEdit:focus {{
                border: 1px solid {WIN11_ACCENT};
            }}
            QListWidget {{
                background-color: {WIN11_SURFACE};
                border-radius: 12px;
                padding: 6px;
            }}
            QListWidget::item:selected {{
                background-color: {WIN11_ACCENT};
                border-radius: 8px;
            }}
            QPushButton {{
                background-color: #2a3441;
                border-radius: 10px;
                padding: 8px 14px;
                border: 1px solid #374151;
            }}
            QPushButton:hover {{
                background-color: #334155;
            }}
            QPushButton#accent-button {{
                background-color: {WIN11_ACCENT};
                color: #ffffff;
                border: none;
                font-weight: 600;
            }}
            QPushButton#accent-button:hover {{
                background-color: #1d4ed8;
            }}
            QCheckBox {{
                spacing: 8px;
                color: {WIN11_TEXT};
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 2px solid #4b5563;
                background: {WIN11_SURFACE};
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {WIN11_ACCENT};
            }}
            QCheckBox::indicator:checked {{
                background-color: {WIN11_ACCENT};
                border: 2px solid {WIN11_ACCENT};
                image: none;
            }}
            QCheckBox::indicator:checked:disabled {{
                background-color: #374151;
                border: 2px solid #4b5563;
            }}
            QPlainTextEdit {{
                background-color: {WIN11_SURFACE};
                border-radius: 12px;
                border: 1px solid #374151;
                padding: 10px;
            }}
            QLabel {{
                font-size: 14px;
            }}
        """
        app.setStyleSheet(stylesheet)

    def _append_log(self, message: str) -> None:
        self._log_output.appendPlainText(message)
        self._log_output.verticalScrollBar().setValue(
            self._log_output.verticalScrollBar().maximum()
        )

    def _handle_method_selection(self) -> None:
        items = self._method_list.selectedItems()
        if not items:
            self._status_label.setText("Select a measurement to configure.")
            return
        method_name = items[0].text()
        method = self._measure_methods_map[method_name]
        self._form_container.refresh_instrument_fields()
        self._form_container.set_method(method)
        doc = inspect.getdoc(method) or ""
        self._status_label.setText(doc.splitlines()[0] if doc else "Parameters ready.")

    def _browse_custom_path(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select out_db_path directory",
            "",
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
        )
        if directory:
            self._custom_db_input.setText(directory)

    def _run_measurement(self) -> None:
        try:
            if self._active_thread and self._active_thread.is_alive():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Measurement Running",
                    "A measurement is already running. Please wait for it to complete.",
                )
                return

            items = self._method_list.selectedItems()
            if not items:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Measurement Selected",
                    "Select a measurement method from the list before running.",
                )
                return
            method_name = items[0].text()

            measure_flow = self._prepare_measure_flow()
            if measure_flow is None:
                return

            try:
                kwargs = self._gather_parameters()
            except ValueError as exc:
                QtWidgets.QMessageBox.critical(self, "Invalid Parameters", str(exc))
                return

            measure_method = getattr(measure_flow, method_name)
            self._status_label.setText(f"Running {method_name}...")
            self._run_button.setEnabled(False)

            self._active_thread = MeasureRunner(
                method_name,
                measure_method,
                kwargs,
                on_finished=self._handle_finished,
                on_error=self._handle_error,
                error_log_path=self._error_log_path,
            )
            self._active_thread.start()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover  # pylint: disable=broad-except
            error_traceback = traceback.format_exc()
            log_file = self._save_error_log(
                error_traceback, f"Failed to start measurement: {exc}"
            )
            logging.getLogger(__name__).exception(
                "Unexpected error while starting measurement."
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Measurement Error",
                f"An unexpected error occurred while starting the measurement:\n{exc}\n\n"
                f"Error log saved to:\n{log_file}",
            )
            self._status_label.setText("Measurement start failed.")
            self._run_button.setEnabled(True)
            self._active_thread = None

    def _gather_parameters(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for name, parameter in self._form_container.parameters.items():
            value, is_empty, needs_parse = self._form_container.extract_value(name)
            if is_empty:
                if parameter.default is SIGNATURE_EMPTY:
                    raise ValueError(f"Parameter '{name}' is required.")
                continue
            if needs_parse and isinstance(value, str):
                value = self._parse_input(value)
            kwargs[name] = value
        if "use_dash" in self._form_container.parameters and "use_dash" not in kwargs:
            kwargs["use_dash"] = True
        return kwargs

    @staticmethod
    def _parse_input(raw_value: str) -> Any:
        lowered = raw_value.lower()
        if lowered == "none":
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return raw_value

    def _cleanup_after_measurement(self, force_recreate: bool = False) -> None:
        """Clean up resources after measurement completion or error.
        
        Args:
            force_recreate: If True, force recreation of MeasureFlow instance.
                          If False, just clean dataframes while keeping instruments loaded.
        """
        try:
            if self._measure_flow is not None:
                if hasattr(self._measure_flow, "dfs"):
                    self._measure_flow.dfs.clear()
                    import pandas as pd
                    self._measure_flow.dfs["curr_measure"] = pd.DataFrame()
                    logging.getLogger(__name__).info("Cleared and reset measurement dataframes.")
                
                if force_recreate:
                    self._measure_flow = None
                    self._active_project_name = None
                    self._active_out_db_path = None
                    logging.getLogger(__name__).info("MeasureFlow instance will be recreated for next run.")
                    self._append_log("Full cleanup: MeasureFlow will be recreated. Instruments need reloading.")
                else:
                    self._append_log("Measurement cleanup completed. Instruments preserved, ready for next run.")
            else:
                self._append_log("No active MeasureFlow to clean up.")
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("Error during measurement cleanup.")
            self._append_log(f"Warning: Cleanup encountered issue: {exc}")

    def _handle_finished(self, method_name: str) -> None:
        self._cleanup_after_measurement()
        self._status_label.setText(f"{method_name} completed successfully. Ready for next measurement.")
        self._run_button.setEnabled(True)
        self._active_thread = None

    def _handle_error(self, method_name: str, exc: Exception, log_file: str) -> None:
        self._cleanup_after_measurement(force_recreate=False)
        
        logging.getLogger(__name__).exception(
            "Measurement '%s' encountered an error.", method_name
        )
        self._status_label.setText(f"{method_name} failed. Ready for next measurement.")
        self._run_button.setEnabled(True)
        self._active_thread = None

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Measurement Error")
        msg_box.setText(f"{method_name} failed with error:\n\n{type(exc).__name__}: {exc}")
        msg_box.setInformativeText(
            f"Error log saved to:\n{log_file}\n\n"
            "Cleanup completed. You can:\n"
            "• OK: Continue with current setup\n"
            "• Discard: Force full cleanup (recreate MeasureFlow, instruments need reloading)\n"
            "• Reset: Reset GUI to initial state"
        )
        msg_box.setDetailedText(traceback.format_exc())
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Reset
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)

        result = msg_box.exec()

        if result == QtWidgets.QMessageBox.StandardButton.Discard:
            self._cleanup_after_measurement(force_recreate=True)
        elif result == QtWidgets.QMessageBox.StandardButton.Reset:
            self._reset_to_initial_state()

        self._append_log(f"ERROR: {method_name} failed - {exc}")
        self._append_log(f"Error log saved to: {log_file}")

    def _manual_cleanup(self) -> None:
        """Manually trigger cleanup of measurement resources."""
        try:
            if self._active_thread and self._active_thread.is_alive():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Measurement Running",
                    "Cannot perform cleanup while a measurement is running. "
                    "Please wait for it to complete or restart the application.",
                )
                return

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Force Cleanup")
            msg_box.setText("Choose cleanup level:")
            msg_box.setInformativeText(
                "• Light Cleanup: Clear dataframes only (keeps instruments loaded)\n"
                "• Full Cleanup: Recreate MeasureFlow (instruments need reloading)"
            )
            light_button = msg_box.addButton("Light Cleanup", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            full_button = msg_box.addButton("Full Cleanup", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            msg_box.setDefaultButton(light_button)

            msg_box.exec()
            clicked = msg_box.clickedButton()

            if clicked == cancel_button:
                return
            elif clicked == full_button:
                self._cleanup_after_measurement(force_recreate=True)
                self._status_label.setText("Full cleanup completed. MeasureFlow will be recreated.")
            else:  # light_button
                self._cleanup_after_measurement(force_recreate=False)
                self._status_label.setText("Light cleanup completed. Ready for next measurement.")

        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("Error during manual cleanup.")
            QtWidgets.QMessageBox.critical(
                self,
                "Cleanup Error",
                f"An error occurred during cleanup:\n{exc}",
            )

    def _open_dash_in_browser(self) -> None:
        webbrowser.open(f"http://127.0.0.1:{self._dash_bridge.port}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    dash_bridge = DashBridge()
    window = MeasureFlowGui(dash_bridge)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

