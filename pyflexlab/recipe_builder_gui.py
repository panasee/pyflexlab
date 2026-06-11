"""
Minimal PyQt recipe builder for the new measurement flow.

This module is intentionally separate from the older pyflexlab.gui module.  The
GUI edits a structured recipe specification; translating that specification
into a MeasurementRecipe is kept as a future, explicit step.
"""

from __future__ import annotations

import json
import math
import socket
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Literal


ModuleCategory = Literal["source", "sense", "external", "plot"]

# The GUI has four visual boxes.  This mapping is the single place that ties the
# public category name stored on each module to the attribute used by
# GuiRecipeSpec.  Keeping this explicit makes category validation easy to audit.
CATEGORY_ATTRS: dict[ModuleCategory, str] = {
    "source": "sources",
    "sense": "senses",
    "external": "externals",
    "plot": "plots",
}

# Custom MIME type used for internal drag/drop payloads.  Qt drag/drop works by
# moving opaque bytes through QMimeData; this value prevents our module payloads
# from being confused with plain text, files, or other draggable data.
MIME_TYPE = "application/x-pyflexlab-module"

LIVE_PLOT_MODULE_IDS = frozenset({"plot.vi_curve", "plot.rh_loop"})


def _find_free_local_port() -> int:
    """Reserve a free localhost TCP port for the embedded Dash preview."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True, slots=True)
class ModuleDefinition:
    """
    A reusable module shown in the left-side GUI module library.

    Definitions are catalog entries, not selected experiment steps.  A user can
    drag the same definition into a box multiple times; each drop becomes a
    ModuleInstance with its own editable parameters.
    """

    module_id: str
    label: str
    category: ModuleCategory
    description: str = ""
    default_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the definition for Qt item storage and drag/drop payloads."""
        return {
            "module_id": self.module_id,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "default_parameters": self.default_parameters,
        }


@dataclass(frozen=True, slots=True)
class ModuleInstance:
    """
    A module instance dropped into one of the recipe boxes.

    Instances represent the user's current recipe draft.  They intentionally do
    not know how to call instruments or build MeasurementRecipe objects yet;
    that translation layer should stay separate from the GUI editing layer.
    """

    module_id: str
    label: str
    category: ModuleCategory
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_definition(cls, definition: ModuleDefinition) -> "ModuleInstance":
        """Create an editable dropped module from an immutable library entry."""
        return cls(
            module_id=definition.module_id,
            label=definition.label,
            category=definition.category,
            parameters=dict(definition.default_parameters),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleInstance":
        """
        Rehydrate an instance from Qt item data or saved JSON.

        Older drag payloads may still contain default_parameters rather than
        parameters, so both keys are accepted.  Unknown categories still fail
        loudly through _validate_category.
        """
        return cls(
            module_id=str(data["module_id"]),
            label=str(data["label"]),
            category=_validate_category(str(data["category"])),
            parameters=dict(data.get("parameters", data.get("default_parameters", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON shape used by GuiRecipeSpec.to_json()."""
        return {
            "module_id": self.module_id,
            "label": self.label,
            "category": self.category,
            "parameters": self.parameters,
        }

    def with_parameters(self, parameters: dict[str, Any]) -> "ModuleInstance":
        """Return a copy with edited parameters."""
        return ModuleInstance(
            module_id=self.module_id,
            label=self.label,
            category=self.category,
            parameters=dict(parameters),
        )

    def display_label(self) -> str:
        """Compact label for list widgets that hints when parameters are set."""
        if not self.parameters:
            return self.label
        return f"{self.label}  {json.dumps(self.parameters, sort_keys=True)}"


@dataclass(slots=True)
class GuiRecipeSpec:
    """
    Structured GUI-side recipe specification.

    This is the editable output of the builder.  It is deliberately less
    powerful than MeasurementRecipe: it says which logical modules the user
    selected and stores GUI parameters, while later code can decide whether that
    selection is valid for a real measurement workflow.
    """

    sources: list[ModuleInstance] = field(default_factory=list)
    senses: list[ModuleInstance] = field(default_factory=list)
    externals: list[ModuleInstance] = field(default_factory=list)
    plots: list[ModuleInstance] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    _BOX_CATEGORIES: ClassVar[tuple[tuple[str, ModuleCategory], ...]] = (
        ("sources", "source"),
        ("senses", "sense"),
        ("externals", "external"),
        ("plots", "plot"),
    )

    def __post_init__(self) -> None:
        # Validate box/category consistency immediately.  A source module in the
        # sense box is a user or serialization error; do not silently coerce it.
        for attr_name, expected_category in self._BOX_CATEGORIES:
            for module in getattr(self, attr_name):
                if module.category != expected_category:
                    raise ValueError(
                        f"{module.category} module cannot be added to {expected_category}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete GUI recipe spec without Qt-specific objects."""
        return {
            "sources": [module.to_dict() for module in self.sources],
            "senses": [module.to_dict() for module in self.senses],
            "externals": [module.to_dict() for module in self.externals],
            "plots": [module.to_dict() for module in self.plots],
            "parameters": self.parameters,
        }

    def to_json(self) -> str:
        """Pretty-print the current spec for the right-side preview panel."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def _modules_for_category(self, category: ModuleCategory) -> list[ModuleInstance]:
        return getattr(self, CATEGORY_ATTRS[category])

    def move_module(
        self, category: ModuleCategory, from_index: int, to_index: int
    ) -> None:
        """Move one module within its category list while preserving category order."""
        modules = self._modules_for_category(category)
        if from_index == to_index:
            return
        module = modules.pop(from_index)
        modules.insert(to_index, module)

    def update_module_parameters(
        self, category: ModuleCategory, index: int, parameters: dict[str, Any]
    ) -> None:
        """Update one dropped module's editable parameter dict."""
        modules = self._modules_for_category(category)
        modules[index] = modules[index].with_parameters(parameters)


def _spec_has_live_plot(spec: GuiRecipeSpec) -> bool:
    """Return whether the GUI spec asks for an active live plot."""
    return any(module.module_id in LIVE_PLOT_MODULE_IDS for module in spec.plots)


def _parse_parameters_json(raw: str) -> dict[str, Any]:
    """Parse the parameter editor content and require a JSON object."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("parameters JSON must be an object")
    return data


class _RecipeDashPreview:
    """Small Dash/Plotly preview used by the standalone recipe builder."""

    def __init__(self, port: int | None = None) -> None:
        self.port = port if port is not None else _find_free_local_port()
        self._lock = threading.Lock()
        self._live_plot_enabled = False
        self._started = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def update_from_spec(self, spec: GuiRecipeSpec) -> None:
        with self._lock:
            self._live_plot_enabled = _spec_has_live_plot(spec)

    def _is_live_plot_enabled(self) -> bool:
        with self._lock:
            return self._live_plot_enabled

    def start(self) -> None:
        """
        Start the local Dash server in a daemon thread.

        Dash/Plotly are imported here so the data model remains importable in
        non-GUI and test environments.
        """
        if self._started:
            return

        try:
            from dash import Dash, Input, Output, dcc, html
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover - optional runtime deps
            raise RuntimeError(
                "Dash and Plotly are required for the recipe plot preview."
            ) from exc

        app = Dash(__name__)
        app.layout = html.Div(
            [
                html.Div(
                    "Recipe Plot Preview",
                    style={
                        "fontFamily": "Arial, sans-serif",
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "margin": "8px 12px 0",
                    },
                ),
                dcc.Graph(id="recipe-live-plot", style={"height": "360px"}),
                dcc.Interval(
                    id="recipe-live-plot-timer",
                    interval=1000,
                    n_intervals=0,
                ),
            ],
            style={"backgroundColor": "#ffffff", "height": "100vh"},
        )

        @app.callback(
            Output("recipe-live-plot", "figure"),
            Input("recipe-live-plot-timer", "n_intervals"),
        )
        def _update_plot(n_intervals: int) -> Any:
            live_plot_enabled = self._is_live_plot_enabled()
            fig = go.Figure()
            if live_plot_enabled:
                count = min(max(n_intervals + 1, 2), 120)
                start = max(0, n_intervals - count + 1)
                x_values = list(range(start, start + count))
                y_values = [
                    math.sin(value / 6) + 0.15 * math.cos(value / 2)
                    for value in x_values
                ]
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name="live_xy",
                    )
                )
                title = "Live XY Plot"
            else:
                title = "Drop Live XY Plot into the Plot box"
            fig.update_layout(
                title=title,
                margin={"l": 48, "r": 24, "t": 48, "b": 48},
                xaxis_title="sample",
                yaxis_title="value",
                template="plotly_white",
                uirevision="recipe-preview",
            )
            return fig

        thread = threading.Thread(
            target=app.run,
            kwargs={
                "host": "127.0.0.1",
                "port": self.port,
                "debug": False,
                "use_reloader": False,
            },
            daemon=True,
            name="recipe-dash-preview",
        )
        thread.start()
        self._started = True


# Initial module library for the prototype UI.  These are intentionally broad
# logical roles rather than concrete driver classes; the future translator can
# map them to MeasureManager/get_measure_dict inputs once the GUI contract is
# settled.
DEFAULT_MODULE_LIBRARY: tuple[ModuleDefinition, ...] = (
    ModuleDefinition(
        module_id="source.fixed_voltage",
        label="Fixed Voltage Source",
        category="source",
        description="Fixed DC voltage source provided by a source meter.",
        default_parameters={
            "value": 0,
            "high": 0,
            "low": 0,
            "meter": "",
            "compliance": 1,
            "freq": None,
        },
    ),
    ModuleDefinition(
        module_id="source.fixed_current",
        label="Fixed Current Source",
        category="source",
        description="Fixed DC or AC current source provided by a source meter.",
        default_parameters={
            "value": 0,
            "high": 0,
            "low": 0,
            "meter": "",
            "compliance": 1,
            "freq": None,
        },
    ),
    ModuleDefinition(
        module_id="source.sweep_voltage",
        label="Sweep Voltage Source",
        category="source",
        description="DC or AC voltage sweep provided by a source meter.",
        default_parameters={
            "max_value": 0,
            "step_value": 0,
            "high": 0,
            "low": 0,
            "sweepmode": "0-max-0",
            "meter": "",
            "compliance": 1,
            "freq": None,
        },
    ),
    ModuleDefinition(
        module_id="source.sweep_current",
        label="Sweep Current Source",
        category="source",
        description="DC or AC current sweep provided by a source meter.",
        default_parameters={
            "max_value": 0,
            "step_value": 0,
            "high": 0,
            "low": 0,
            "sweepmode": "0-max-0",
            "meter": "",
            "compliance": 1,
            "freq": None,
        },
    ),
    ModuleDefinition(
        module_id="sense.voltage",
        label="Voltage Sense",
        category="sense",
        description="DC or AC voltage readback from a meter or source meter.",
        default_parameters={
            "high": 0,
            "low": 0,
            "comment": "",
            "meter": "",
            "ac_dc": "dc",
        },
    ),
    ModuleDefinition(
        module_id="sense.current",
        label="Current Sense",
        category="sense",
        description="DC or AC current readback from a meter or source meter.",
        default_parameters={
            "high": 0,
            "low": 0,
            "comment": "",
            "meter": "",
            "ac_dc": "dc",
        },
    ),
    ModuleDefinition(
        module_id="external.fixed_magnetic_field",
        label="Fixed Magnetic Field",
        category="external",
        description="Fixed external magnet field.",
        default_parameters={"value": 0},
    ),
    ModuleDefinition(
        module_id="external.vary_magnetic_field",
        label="Vary Magnetic Field",
        category="external",
        description="External magnet field ramp or loop.",
        default_parameters={"start": 0, "stop": 0},
    ),
    ModuleDefinition(
        module_id="external.sweep_magnetic_field",
        label="Sweep Magnetic Field",
        category="external",
        description="External magnet field sweep.",
        default_parameters={
            "start": 0,
            "stop": 0,
            "step": 0,
            "sweepmode": "0-max-0",
        },
    ),
    ModuleDefinition(
        module_id="external.fixed_temperature",
        label="Fixed Temperature",
        category="external",
        description="Fixed temperature controller target.",
        default_parameters={"value": 0},
    ),
    ModuleDefinition(
        module_id="external.vary_temperature",
        label="Vary Temperature",
        category="external",
        description="Temperature controller vary range.",
        default_parameters={"start": 0, "stop": 0},
    ),
    ModuleDefinition(
        module_id="external.sweep_temperature",
        label="Sweep Temperature",
        category="external",
        description="Temperature controller sweep.",
        default_parameters={
            "start": 0,
            "stop": 0,
            "step": 0,
            "sweepmode": "0-max-0",
        },
    ),
    ModuleDefinition(
        module_id="external.fixed_angle",
        label="Fixed Angle",
        category="external",
        description="Fixed rotator angle.",
        default_parameters={"value": 0},
    ),
    ModuleDefinition(
        module_id="external.vary_angle",
        label="Vary Angle",
        category="external",
        description="Rotator angle vary range.",
        default_parameters={"start": 0, "stop": 0},
    ),
    ModuleDefinition(
        module_id="external.sweep_angle",
        label="Sweep Angle",
        category="external",
        description="Rotator angle sweep.",
        default_parameters={
            "start": 0,
            "stop": 0,
            "step": 0,
            "sweepmode": "0-max-0",
        },
    ),
    ModuleDefinition(
        module_id="plot.vi_curve",
        label="V-I Curve Plot",
        category="plot",
        description="Live V-I curve plot using mapped PlotSeries columns.",
        default_parameters={"saving_interval": 7},
    ),
    ModuleDefinition(
        module_id="plot.rh_loop",
        label="R-H Loop Plot",
        category="plot",
        description="Live R-H loop plot using mapped PlotSeries columns.",
        default_parameters={"saving_interval": 7},
    ),
    ModuleDefinition(
        module_id="plot.record_only",
        label="Record Only",
        category="plot",
        description="Record data without enabling live plotting.",
    ),
)


def _validate_category(value: str) -> ModuleCategory:
    """Validate untrusted JSON/category strings before treating them as types."""
    if value not in CATEGORY_ATTRS:
        raise ValueError(f"unknown module category: {value}")
    return value  # type: ignore[return-value]


def _require_pyqt6() -> tuple[Any, Any, Any]:
    """
    Import PyQt6 only when the GUI is launched.

    The data model above is useful in tests and non-GUI tooling, so importing
    this module should not require the optional GUI extra.
    """
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PyQt6 is required for recipe_builder_gui. "
            "Install pyflexlab with the gui extra before launching it."
        ) from exc
    return QtCore, QtGui, QtWidgets


def _module_definition_from_json(raw: bytes) -> ModuleDefinition:
    """Decode the drag/drop JSON payload back into a ModuleDefinition."""
    data = json.loads(raw.decode("utf-8"))
    return ModuleDefinition(
        module_id=str(data["module_id"]),
        label=str(data["label"]),
        category=_validate_category(str(data["category"])),
        description=str(data.get("description", "")),
        default_parameters=dict(data.get("default_parameters", {})),
    )


def launch(
    module_library: Iterable[ModuleDefinition] = DEFAULT_MODULE_LIBRARY,
) -> int:
    """Launch the standalone PyQt recipe builder."""

    QtCore, QtGui, QtWidgets = _require_pyqt6()
    try:
        from PyQt6 import QtWebEngineWidgets
    except ImportError:  # pragma: no cover - optional GUI extra
        QtWebEngineWidgets = None

    dash_preview: _RecipeDashPreview | None = _RecipeDashPreview()
    dash_error = ""
    try:
        dash_preview.start()
    except RuntimeError as exc:  # pragma: no cover - depends on optional deps
        dash_preview = None
        dash_error = str(exc)

    # The Qt widget classes live inside launch() so importing this module stays
    # PyQt-free until a user explicitly starts the GUI.
    class ModuleLibraryList(QtWidgets.QListWidget):
        """Left-side list of draggable module definitions."""

        def __init__(self, modules: Iterable[ModuleDefinition]) -> None:
            super().__init__()
            self.setDragEnabled(True)
            self.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
            for module in modules:
                item = QtWidgets.QListWidgetItem(module.label)
                item.setToolTip(module.description)
                # Store a plain dict on the item instead of the dataclass itself.
                # Plain data survives Qt's QVariant conversion more predictably.
                item.setData(QtCore.Qt.ItemDataRole.UserRole, module.to_dict())
                self.addItem(item)

        def mimeTypes(self) -> list[str]:
            """Advertise the one internal payload format this list can drag."""
            return [MIME_TYPE]

        def mimeData(self, items: list[Any]) -> Any:
            """Package the selected module definition as JSON bytes for dragging."""
            mime_data = QtCore.QMimeData()
            if not items:
                return mime_data
            module_data = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            payload = json.dumps(module_data).encode("utf-8")
            mime_data.setData(MIME_TYPE, QtCore.QByteArray(payload))
            return mime_data

        def supportedDropActions(self) -> Any:
            # Dragging from the library should copy a definition into a box, not
            # remove the original definition from the module library.
            return QtCore.Qt.DropAction.CopyAction

    class ModuleDropList(QtWidgets.QListWidget):
        """One category-specific target box in the center of the window."""

        def __init__(self, category: ModuleCategory, on_changed: Any) -> None:
            super().__init__()
            self.category = category
            # The parent window passes a preview refresh callback.  The drop list
            # should not know about the preview widget directly.
            self._on_changed = on_changed
            self.setAcceptDrops(True)
            self.setDragEnabled(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
            self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
            self.setDropIndicatorShown(True)
            self.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )

        def dragEnterEvent(self, event: Any) -> None:
            """Allow the drag only if the payload belongs in this box."""
            self._accept_matching_module(event)

        def dragMoveEvent(self, event: Any) -> None:
            """Keep rejecting wrong-category modules while the cursor moves."""
            self._accept_matching_module(event)

        def dropEvent(self, event: Any) -> None:
            """Convert an accepted module definition into a module instance."""
            if not event.mimeData().hasFormat(MIME_TYPE):
                super().dropEvent(event)
                self._on_changed()
                return

            definition = _module_definition_from_json(
                bytes(event.mimeData().data(MIME_TYPE))
            )
            if definition.category != self.category:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Wrong module box",
                    f"{definition.label} belongs in the {definition.category} box.",
                )
                event.ignore()
                return

            self.add_module(ModuleInstance.from_definition(definition))
            event.acceptProposedAction()
            # Keep the preview synchronized after every user-visible change.
            self._on_changed()

        def add_module(self, module: ModuleInstance) -> None:
            """Add a dropped/loaded instance to this visual box."""
            item = QtWidgets.QListWidgetItem(module.display_label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, module.to_dict())
            self.addItem(item)

        def edit_selected(self) -> None:
            """Edit the selected module parameters as a JSON object."""
            selected_items = self.selectedItems()
            if not selected_items:
                return

            item = selected_items[0]
            module = ModuleInstance.from_dict(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
            )
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"Edit {module.label} Parameters")
            layout = QtWidgets.QVBoxLayout(dialog)
            editor = QtWidgets.QPlainTextEdit()
            editor.setPlainText(json.dumps(module.parameters, indent=2, sort_keys=True))
            editor.setFont(QtGui.QFont("Consolas", 10))
            layout.addWidget(editor)

            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            try:
                parameters = _parse_parameters_json(editor.toPlainText())
            except (json.JSONDecodeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid parameters",
                    str(exc),
                )
                return

            edited_module = module.with_parameters(parameters)
            item.setText(edited_module.display_label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, edited_module.to_dict())
            self._on_changed()

        def move_selected(self, delta: int) -> None:
            """Move selected module up or down within this category box."""
            selected_items = self.selectedItems()
            if not selected_items:
                return
            item = selected_items[0]
            row = self.row(item)
            new_row = row + delta
            if new_row < 0 or new_row >= self.count():
                return
            moved_item = self.takeItem(row)
            self.insertItem(new_row, moved_item)
            self.setCurrentItem(moved_item)
            self._on_changed()

        def remove_selected(self) -> None:
            """Remove selected dropped modules and refresh the preview."""
            for item in self.selectedItems():
                self.takeItem(self.row(item))
            self._on_changed()

        def instances(self) -> list[ModuleInstance]:
            """Return all current box contents as plain ModuleInstance objects."""
            modules: list[ModuleInstance] = []
            for row in range(self.count()):
                item = self.item(row)
                modules.append(
                    ModuleInstance.from_dict(
                        item.data(QtCore.Qt.ItemDataRole.UserRole)
                    )
                )
            return modules

        def _accept_matching_module(self, event: Any) -> None:
            """
            Shared category gate for drag-enter and drag-move events.

            Rejecting here gives immediate visual feedback and prevents the user
            from dropping source modules into sense/external/plot boxes.
            """
            if not event.mimeData().hasFormat(MIME_TYPE):
                if event.source() is self:
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return
            definition = _module_definition_from_json(
                bytes(event.mimeData().data(MIME_TYPE))
            )
            if definition.category == self.category:
                event.acceptProposedAction()
            else:
                event.ignore()

    class RecipeBuilderWindow(QtWidgets.QMainWindow):
        """Main standalone builder window."""

        def __init__(self, modules: Iterable[ModuleDefinition]) -> None:
            super().__init__()
            self.setWindowTitle("PyFlexLab Recipe Builder")
            self.resize(1180, 720)
            # Store the drop widgets by category so _current_spec can assemble a
            # GuiRecipeSpec without depending on layout positions.
            self._drop_lists: dict[ModuleCategory, ModuleDropList] = {}
            self._dash_preview = dash_preview
            self._build_ui(modules)
            self._refresh_preview()

        def _build_ui(self, modules: Iterable[ModuleDefinition]) -> None:
            """Build the three-column UI: library, drop boxes, JSON preview."""
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root_layout = QtWidgets.QHBoxLayout(central)

            library_group = QtWidgets.QGroupBox("Module Library")
            library_layout = QtWidgets.QVBoxLayout(library_group)
            library_layout.addWidget(ModuleLibraryList(modules))
            root_layout.addWidget(library_group, 1)

            boxes_widget = QtWidgets.QWidget()
            boxes_layout = QtWidgets.QGridLayout(boxes_widget)
            for index, (category, title) in enumerate(
                (
                    ("source", "Source"),
                    ("sense", "Sense"),
                    ("external", "External"),
                    ("plot", "Plot"),
                )
            ):
                # Four independent category boxes.  There is no ordering or graph
                # semantics between boxes in this prototype, but the item order
                # inside each box is the get_measure_dict order for that category.
                group = QtWidgets.QGroupBox(title)
                layout = QtWidgets.QVBoxLayout(group)
                drop_list = ModuleDropList(category, self._refresh_preview)
                self._drop_lists[category] = drop_list
                layout.addWidget(drop_list)

                button_layout = QtWidgets.QHBoxLayout()
                edit_button = QtWidgets.QPushButton("Edit Parameters")
                edit_button.clicked.connect(drop_list.edit_selected)
                button_layout.addWidget(edit_button)

                up_button = QtWidgets.QPushButton("Move Up")
                up_button.clicked.connect(
                    lambda _, item_list=drop_list: item_list.move_selected(-1)
                )
                button_layout.addWidget(up_button)

                down_button = QtWidgets.QPushButton("Move Down")
                down_button.clicked.connect(
                    lambda _, item_list=drop_list: item_list.move_selected(1)
                )
                button_layout.addWidget(down_button)

                remove_button = QtWidgets.QPushButton("Remove")
                remove_button.clicked.connect(drop_list.remove_selected)
                button_layout.addWidget(remove_button)
                layout.addLayout(button_layout)

                boxes_layout.addWidget(group, index // 2, index % 2)
            root_layout.addWidget(boxes_widget, 2)

            preview_group = QtWidgets.QGroupBox("Recipe Spec Preview")
            preview_layout = QtWidgets.QVBoxLayout(preview_group)
            self._preview = QtWidgets.QPlainTextEdit()
            self._preview.setReadOnly(True)
            self._preview.setFont(QtGui.QFont("Consolas", 10))
            preview_layout.addWidget(self._preview)

            refresh_button = QtWidgets.QPushButton("Refresh Preview")
            refresh_button.clicked.connect(self._refresh_preview)
            preview_layout.addWidget(refresh_button)
            root_layout.addWidget(preview_group, 2)

            plot_group = QtWidgets.QGroupBox("Dash Plot")
            plot_layout = QtWidgets.QVBoxLayout(plot_group)
            plot_header_layout = QtWidgets.QHBoxLayout()
            open_browser_button = QtWidgets.QPushButton("Open in Browser")
            open_browser_button.setEnabled(self._dash_preview is not None)
            open_browser_button.clicked.connect(self._open_dash_in_browser)
            plot_header_layout.addStretch()
            plot_header_layout.addWidget(open_browser_button)
            plot_layout.addLayout(plot_header_layout)
            if self._dash_preview is None:
                error_label = QtWidgets.QLabel(dash_error)
                error_label.setWordWrap(True)
                error_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                plot_layout.addWidget(error_label)
            elif QtWebEngineWidgets is None:
                info_label = QtWidgets.QLabel(
                    "PyQt6-WebEngine is not installed. "
                    "Use Open in Browser to view the Dash plot."
                )
                info_label.setWordWrap(True)
                info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                plot_layout.addWidget(info_label)
            else:
                dash_view = QtWebEngineWidgets.QWebEngineView()
                dash_view.setMinimumHeight(260)
                dash_view.setUrl(QtCore.QUrl(self._dash_preview.url))
                plot_layout.addWidget(dash_view)
            root_layout.addWidget(plot_group, 2)

        def _current_spec(self) -> GuiRecipeSpec:
            """Collect all drop-box contents into a validated GUI spec."""
            return GuiRecipeSpec(
                sources=self._drop_lists["source"].instances(),
                senses=self._drop_lists["sense"].instances(),
                externals=self._drop_lists["external"].instances(),
                plots=self._drop_lists["plot"].instances(),
            )

        def _refresh_preview(self) -> None:
            """Render the current GUI spec as JSON in the preview pane."""
            spec = self._current_spec()
            self._preview.setPlainText(spec.to_json())
            if self._dash_preview is not None:
                self._dash_preview.update_from_spec(spec)

        def _open_dash_in_browser(self) -> None:
            """Open the local Dash preview in the user's default browser."""
            if self._dash_preview is None:
                return
            webbrowser.open(self._dash_preview.url)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Keep a local reference to the window until app.exec() returns.  Without
    # this, Python could garbage-collect the window after launch().
    window = RecipeBuilderWindow(module_library)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main() -> None:
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
