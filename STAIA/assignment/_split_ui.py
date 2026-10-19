"""
Interactive Jupyter widget UI for manually splitting spatial samples.

This module is intentionally kept separate from the core logic in
the assignment module.  It handles all ipywidgets / matplotlib concerns and
manages the mutable split-session state (history, pending preview) in a
local object rather than in global variables.

Split types
-----------
- vertical / horizontal: a single straight cut at a given x or y value.
- diagonal: a single straight cut through two user-given points.
- freehand: a hand-drawn, possibly curved cut — useful for samples that
  aren't neatly separated by a straight line. Click "Start drawing" and
  drag the mouse across the plot; release to finish, then Preview as usual.

Backend requirement for freehand drawing
-----------------------------------------
Freehand drawing needs a *live* matplotlib backend that can forward mouse
events to Python — the static inline backend cannot do this. Before
creating a SplitSession, run once in a notebook cell:

>>> %matplotlib widget

This requires the ``ipympl`` package (``pip install ipympl``); restart the
kernel after installing if it was just added. This works fine over a
remote JupyterHub/HPC connection since all interactivity goes over the
same browser/websocket connection as the rest of the notebook — no X11
forwarding is needed. If you only use vertical/horizontal/diagonal
splits, the default inline backend continues to work unchanged.

Typical usage (in a Jupyter notebook)
--------------------------------------
>>> %matplotlib widget
>>> from split_ui import SplitSession
>>> session = SplitSession(df_base)   # df_base has columns x, y, sample_id
>>> session.show()                    # renders the interactive widget
>>>
>>> # After you are done splitting:
>>> df_result, ids = session.result()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Import pure logic from the functional API
from ._assignment_module import (
    replay_all_splits,
    _do_one_split,
    _renumber,
    _make_vertical_record,
    _make_horizontal_record,
    _make_diagonal_record,
    _make_freehand_record,
    plot_samples
)


def _draw_cut(ax: plt.Axes, record: dict) -> None:
    """Overlay a split line on *ax* based on a split record."""
    stype = record["type"]
    if stype == "vertical":
        ax.axvline(record["single_value"], color="white", lw=2, ls="--", alpha=0.85)
    elif stype == "horizontal":
        ax.axhline(record["single_value"], color="white", lw=2, ls="--", alpha=0.85)
    elif stype == "diagonal":
        x1, x2 = record["x_points"]
        y1, y2 = record["y_points"]
        ax.plot([x1, x2], [y1, y2], color="white", lw=2, ls="--", alpha=0.85)
        ax.scatter([x1, x2], [y1, y2], color="white", s=50, zorder=6)
    else:  # freehand — an arbitrary-length polyline, not just two points
        xs, ys = record["x_points"], record["y_points"]
        ax.plot(xs, ys, color="white", lw=2, ls="--", alpha=0.85)
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color="white", s=50, zorder=6)


def _render(
    w_out: widgets.Output, df, ids: list[int], record: dict | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Clear *w_out* and draw the current segmentation (with optional cut preview).

    Returns the (fig, ax) pair so callers can attach mouse-event handlers to
    ``fig.canvas`` (used for freehand drawing). Note this intentionally does
    NOT call plt.close(fig) — under the ipympl ("%matplotlib widget") backend
    the figure needs to stay alive for its canvas to remain interactive.
    Callers that create a new figure on top of an old one should close the
    previous figure themselves (see SplitSession._close_active_fig) to avoid
    accumulating open figures over a long session.
    """
    w_out.clear_output(wait=True)
    with w_out:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1e1e2e")
        plot_samples(df, ids, ax=ax)
        if record is not None:
            _draw_cut(ax, record)
        plt.tight_layout()
        plt.show()
    return fig, ax


class SplitSession:
    """
    Manages a single interactive splitting session.

    Parameters
    ----------
    base_df : pd.DataFrame
        The original (pre-split) dataframe. Never mutated.
    """

    def __init__(self, base_df):
        self._base_df = base_df.copy()
        self._history: list[dict] = []
        self._pending = {"record": None, "df": None, "ids": None}
        self._df, self._ids = replay_all_splits(self._base_df, self._history)
        self._w_out     = widgets.Output()
        self._container = widgets.Output()

        # Freehand drawing state
        self._freehand_points: list[tuple[float, float]] = []
        self._freehand_drawing: bool = False
        self._active_fig: plt.Figure | None = None

    def _close_active_fig(self) -> None:
        """Close the previously-rendered figure, if any, before drawing a new one."""
        if self._active_fig is not None:
            try:
                plt.close(self._active_fig)
            except Exception:
                pass
            self._active_fig = None

    def result(self):
        return self._df.copy(), list(self._ids)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def show(self) -> None:
        """
        Display the widget. Safe to call multiple times or from different
        cells — always shows exactly one copy.
        """
        self._container.clear_output(wait=True)
        with self._container:
            display(self._build_controls()) 
        display(self._container)


    def _build_controls(self) -> widgets.VBox:
        """
        Build and RETURN the control widget tree.
        Never calls display() — that is show()'s sole responsibility.
        """
        display(widgets.HTML("""
            <style>
                input[type=number]::-webkit-outer-spin-button,
                input[type=number]::-webkit-inner-spin-button {
                    -webkit-appearance: none;
                    margin: 0;
                }
                input[type=number] {
                    -moz-appearance: textfield;
                }
            </style>
            <script>
                document.addEventListener('wheel', function(e) {
                    if (document.activeElement.type === 'number') {
                        document.activeElement.blur();
                    }
                }, true);
            </script>
        """))
        w_type = widgets.RadioButtons(
            options=["vertical (x)", "horizontal (y)", "diagonal", "freehand"],
            value="vertical (x)",
            description="Split type:",
            style={"description_width": "100px"},
        )
        w_sample = widgets.BoundedIntText(
            value=0, min=0, max=500,
            description="Sample ID:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="220px"),
        )
        w_single_label = widgets.Label("Cut at x =")
        w_single_value = widgets.FloatText(value=5000, layout=widgets.Layout(width="150px"))
        w_single_row   = widgets.HBox([w_single_label, w_single_value])

        w_x1 = widgets.FloatText(value=0,    description="x1 (µm):", style={"description_width": "80px"}, layout=widgets.Layout(width="190px"))
        w_y1 = widgets.FloatText(value=0,    description="y1 (µm):", style={"description_width": "80px"}, layout=widgets.Layout(width="190px"))
        w_x2 = widgets.FloatText(value=1000, description="x2 (µm):", style={"description_width": "80px"}, layout=widgets.Layout(width="190px"))
        w_y2 = widgets.FloatText(value=1000, description="y2 (µm):", style={"description_width": "80px"}, layout=widgets.Layout(width="190px"))
        w_diag_rows = widgets.VBox([widgets.HBox([w_x1, w_y1]), widgets.HBox([w_x2, w_y2])])
        w_diag_rows.layout.display = "none"

        w_freehand_help = widgets.HTML(
            "<small style='color:grey'>Click 'Start drawing', then click-and-drag on the plot "
            "below to trace a (possibly curved) cut line. Release the mouse to finish, then "
            "click Preview. Requires <code>%matplotlib widget</code> — see module docstring.</small>"
        )
        w_btn_draw_start = widgets.Button(description="✏️ Start drawing", button_style="info",
                                           layout=widgets.Layout(width="140px"))
        w_btn_draw_clear = widgets.Button(description="🧹 Clear", layout=widgets.Layout(width="90px"))
        w_draw_status = widgets.HTML("")
        w_freehand_row = widgets.VBox([
            w_freehand_help,
            widgets.HBox([w_btn_draw_start, w_btn_draw_clear, w_draw_status]),
        ])
        w_freehand_row.layout.display = "none"

        w_btn_preview = widgets.Button(description="Preview",     button_style="primary", layout=widgets.Layout(width="120px"))
        w_btn_confirm = widgets.Button(description="✅ Confirm",  button_style="success", layout=widgets.Layout(width="120px"), disabled=True)
        w_btn_discard = widgets.Button(description="❌ Discard",  button_style="danger",  layout=widgets.Layout(width="120px"), disabled=True)
        w_btn_undo    = widgets.Button(description="↩ Undo last", button_style="warning", layout=widgets.Layout(width="120px"))

        w_status = widgets.HTML("")
        w_log    = widgets.HTML("")

        # helpers 
        def _update_log():
            if not self._history:
                w_log.value = "<i style='color:grey'>No splits confirmed yet.</i>"
                return
            lines = ["<b>Confirmed splits:</b><br>"]
            for i, s in enumerate(self._history):
                if s["type"] == "vertical":
                    desc = f"S{s['sample_id']} — vertical at x = {s['single_value']:.1f}"
                elif s["type"] == "horizontal":
                    desc = f"S{s['sample_id']} — horizontal at y = {s['single_value']:.1f}"
                elif s["type"] == "diagonal":
                    desc = (f"S{s['sample_id']} — diagonal "
                            f"({s['x_points'][0]:.0f}, {s['y_points'][0]:.0f}) → "
                            f"({s['x_points'][1]:.0f}, {s['y_points'][1]:.0f})")
                else:  # freehand
                    desc = f"S{s['sample_id']} — freehand cut ({len(s['x_points'])} points)"
                lines.append(f"&nbsp;&nbsp;{i+1}. {desc}<br>")
            w_log.value = "".join(lines)

        def _reset_pending():
            for k in self._pending:
                self._pending[k] = None

        # callbacks 
        def on_type_change(change):
            if change["name"] != "value":
                return
            val = change["new"]
            is_diag     = val == "diagonal"
            is_freehand = val == "freehand"
            w_diag_rows.layout.display     = "" if is_diag else "none"
            w_freehand_row.layout.display  = "" if is_freehand else "none"
            w_single_row.layout.display    = "" if val in ("vertical (x)", "horizontal (y)") else "none"
            if val == "vertical (x)":
                w_single_label.value = "Cut at x ="
            elif val == "horizontal (y)":
                w_single_label.value = "Cut at y ="

        def on_draw_start(_):
            """Show a live plot and start capturing a freehand cut line."""
            self._freehand_points = []
            self._freehand_drawing = False
            w_draw_status.value = ("<span style='color:#f39c12'>Drawing mode: click-and-drag on "
                                   "the plot below, release to finish.</span>")

            df_current, ids_current = replay_all_splits(self._base_df, self._history)
            self._close_active_fig()
            fig, ax = _render(self._w_out, df_current, ids_current, record=None)
            self._active_fig = fig

            def on_press(event):
                if event.inaxes != ax or event.xdata is None:
                    return
                self._freehand_points = [(event.xdata, event.ydata)]
                self._freehand_drawing = True

            def on_move(event):
                if not self._freehand_drawing or event.inaxes != ax or event.xdata is None:
                    return
                self._freehand_points.append((event.xdata, event.ydata))
                if len(self._freehand_points) >= 2:
                    xs, ys = zip(*self._freehand_points[-2:])
                    ax.plot(xs, ys, color="white", lw=2, alpha=0.85)
                    fig.canvas.draw_idle()

            def on_release(event):
                self._freehand_drawing = False
                n = len(self._freehand_points)
                if n >= 2:
                    w_draw_status.value = (f"<span style='color:#2ecc71'>Captured {n} point(s). "
                                           f"Click Preview to see the split.</span>")
                else:
                    w_draw_status.value = ("<span style='color:#e74c3c'>Line too short — "
                                           "click Start drawing and try again.</span>")

            fig.canvas.mpl_connect("button_press_event", on_press)
            fig.canvas.mpl_connect("motion_notify_event", on_move)
            fig.canvas.mpl_connect("button_release_event", on_release)

        def on_draw_clear(_):
            self._freehand_points = []
            w_draw_status.value = "<i style='color:grey'>Drawing cleared.</i>"
            on_draw_start(None)

        def on_preview(_):
            _reset_pending()
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            w_status.value = ""

            split_type_raw    = w_type.value
            sample_id_display = w_sample.value
            df_current, ids_current = replay_all_splits(self._base_df, self._history)

            if sample_id_display not in ids_current:
                w_status.value = (f"<span style='color:#e74c3c'>Sample {sample_id_display} not found. "
                                  f"Available: {ids_current}</span>")
                return

            if split_type_raw == "vertical (x)":
                record = _make_vertical_record(sample_id_display, w_single_value.value)
            elif split_type_raw == "horizontal (y)":
                record = _make_horizontal_record(sample_id_display, w_single_value.value)
            elif split_type_raw == "diagonal":
                record = _make_diagonal_record(sample_id_display, w_x1.value, w_y1.value, w_x2.value, w_y2.value)
            else:  # freehand
                if len(self._freehand_points) < 2:
                    w_status.value = ("<span style='color:#e74c3c'>No freehand line captured yet. "
                                      "Click 'Start drawing' first, then drag across the plot.</span>")
                    return
                xs, ys = zip(*self._freehand_points)
                record = _make_freehand_record(sample_id_display, list(xs), list(ys))

            try:
                df_after = _do_one_split(df_current, sample_id_display, record["type"],
                                        single_value=record.get("single_value"),
                                        x_points=record.get("x_points"),
                                        y_points=record.get("y_points"))
            except ValueError as exc:
                w_status.value = f"<span style='color:#e74c3c'>{exc}</span>"
                return
            df_after, new_ids = _renumber(df_after)
            # new_ids = df_after["sample_id"].unique().tolist()

            self._pending["record"] = record
            self._pending["df"]     = df_after
            self._pending["ids"]    = new_ids

            self._close_active_fig()
            fig, ax = _render(self._w_out, df_after, new_ids, record=record)  # uses self._w_out
            self._active_fig = fig
            w_status.value = (f"<span style='color:#f39c12'>Preview: {len(new_ids)} samples "
                              f"— confirm or discard?</span>")
            w_btn_confirm.disabled = False
            w_btn_discard.disabled = False

        def on_confirm(_):
            if self._pending["record"] is None:
                print("DEBUG: pending is None")
                return
            print(f"DEBUG on_confirm: pending ids={self._pending['ids']}, "
                  f"pending df unique ids={sorted(self._pending['df']['sample_id'].unique())}")
            self._history.append(self._pending["record"])
            self._df  = self._pending["df"]
            self._ids = self._pending["ids"]
            print(f"DEBUG after assign: self._ids={self._ids}")
            w_status.value = (f"<span style='color:#2ecc71'>✅ Confirmed! "
                              f"{len(self._history)} split(s) total, {len(self._ids)} samples.</span>")
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            _update_log()
            self._close_active_fig()
            fig, ax = _render(self._w_out, self._df, self._ids, record=None)
            self._active_fig = fig
            _reset_pending()
            self._freehand_points = []
            w_draw_status.value = ""

        def on_discard(_):
            _reset_pending()
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            w_status.value = "<span style='color:#e74c3c'>❌ Discarded.</span>"
            df_now, ids_now = replay_all_splits(self._base_df, self._history)
            self._close_active_fig()
            fig, ax = _render(self._w_out, df_now, ids_now, record=None)  # uses self._w_out
            self._active_fig = fig
            self._freehand_points = []
            w_draw_status.value = ""

        def on_undo(_):
            if not self._history:
                w_status.value = "<span style='color:#e74c3c'>Nothing to undo.</span>"
                return
            removed = self._history.pop()
            self._df, self._ids = replay_all_splits(self._base_df, self._history)
            w_status.value = (f"<span style='color:#f39c12'>↩ Undid: {removed['type']} on "
                              f"S{removed['sample_id']}. {len(self._history)} split(s) remaining.</span>")
            _update_log()
            self._close_active_fig()
            fig, ax = _render(self._w_out, self._df, self._ids, record=None)  # uses self._w_out
            self._active_fig = fig

        w_type.observe(on_type_change)
        w_btn_preview.on_click(on_preview)
        w_btn_draw_start.on_click(on_draw_start)
        w_btn_draw_clear.on_click(on_draw_clear)
        w_btn_confirm.on_click(on_confirm)
        w_btn_discard.on_click(on_discard)
        w_btn_undo.on_click(on_undo)

        _update_log()
        fig, ax = _render(self._w_out, self._df, self._ids, record=None)  # initial render into self._w_out
        self._active_fig = fig

        return widgets.VBox([
            widgets.HTML("<b>Manual split tool</b><br>"
                         "<small style='color:grey'>Sample IDs in the plot = what you type in 'Sample ID'.</small>"),
            w_type, w_sample, w_single_row, w_diag_rows, w_freehand_row,
            widgets.HBox([w_btn_preview, w_btn_confirm, w_btn_discard, w_btn_undo]),
            w_status, w_log,
            self._w_out,   # the persistent plot output — always the same object
        ])


class LabelSession:
    """
    Interactive UI for assigning experimental labels to spatial samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns ``x``, ``y``, ``sample_id``.
    ids : list[int]
        Sample IDs to label.
    adata : anndata.AnnData, optional
    output_dir : str, optional
    """

    # Namespaced so it never collides with a user's own "sample_id" column.
    # If this column already exists on adata.obs when exporting, STAIA warns
    # and leaves it untouched rather than silently overwriting it.
    OBS_SAMPLE_ID_COL = "sample_id_STAIA"

    def __init__(self, df, ids, adata=None, output_dir="."):
        self._df         = df.copy()
        self._ids        = list(ids)
        self._adata      = adata
        self._output_dir = output_dir
        self._conditions = {cid: f"Sample_{cid}" for cid in self._ids}
        self._w_plot    = widgets.Output()
        self._w_table   = widgets.Output()
        self._container = widgets.Output()

    def result(self) -> dict:
        return dict(self._conditions)

    def show(self) -> None:
        """Display the widget. Safe to call multiple times or from different cells."""
        self._container.clear_output(wait=True)
        with self._container:
            display(self._build_controls()) 
        display(self._container)

    def _build_controls(self) -> widgets.VBox:
        """Build and RETURN the control widget tree. Never calls display()."""
        import pandas as pd
        import os

        w_dropdown = widgets.Dropdown(
            options=[(f"S{cid}  ({(self._df['sample_id'] == cid).sum():,} cells)", cid) for cid in self._ids],
            description="Sample:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="340px"),
        )
        w_condition = widgets.Text(
            value=self._conditions[self._ids[0]],
            placeholder="e.g. treated, control, WT",
            description="Label:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="320px"),
        )
        w_btn_assign = widgets.Button(description="Assign",       button_style="primary", icon="check", layout=widgets.Layout(width="110px"))
        w_btn_export = widgets.Button(description="Save & Export", button_style="success", layout=widgets.Layout(width="160px"))
        w_status     = widgets.HTML(value="")

        # helpers
        def _refresh_plot():
            self._w_plot.clear_output(wait=True)
            with self._w_plot:
                fig, ax = plt.subplots(figsize=(10, 7))
                fig.patch.set_facecolor("#1e1e2e")
                plot_samples(self._df, self._ids, highlight=w_dropdown.value,
                             conditions=self._conditions, ax=ax)
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        def _refresh_table():
            self._w_table.clear_output(wait=True) 
            with self._w_table:
                rows = [(f"S{cid}", self._conditions[cid],
                         f"{(self._df['sample_id'] == cid).sum():,}") for cid in self._ids]
                tbl = pd.DataFrame(rows, columns=["Sample", "Label", "N cells"])
                display(tbl.style.set_properties(**{"text-align": "left"}))

        # callbacks
        def on_dropdown_change(change):
            if change["name"] == "value":
                w_condition.value = self._conditions.get(change["new"], "")
                _refresh_plot()

        def on_assign(_):
            cid = w_dropdown.value
            val = w_condition.value.strip()
            if val:
                self._conditions[cid] = val
                w_status.value = f"<span style='color:#2ecc71'>S{cid} assigned to '{val}'</span>"
                current_index = self._ids.index(cid)
                if current_index < len(self._ids) - 1:
                    w_dropdown.value = self._ids[current_index + 1]
            else:
                w_status.value = "<span style='color:#e74c3c'>Please enter a label name.</span>"
            _refresh_plot()
            _refresh_table()

        def on_export(_):
            w_status.value = "<span style='color:#f39c12'>⏳ Saving...</span>"
            w_btn_export.disabled = True

            # Always (re)derive the sample_id mapping from the *current* session's
            # df, as a local variable first. It's written to adata.obs under a
            # namespaced column name (OBS_SAMPLE_ID_COL, "sample_id_STAIA") so
            # it can never collide with a user's own "sample_id" column. If
            # that namespaced column somehow already exists (e.g. exported
            # once, then the session was re-labelled and re-run without
            # reloading adata), STAIA warns rather than silently overwriting
            # it — the previous behaviour of writing straight to
            # adata.obs["sample_id"] made it impossible to tell whether that
            # column was STAIA's output or pre-existing user data.

            # Prefer a globally unique cell identifier over cell_ID,
            # which is only unique within a FOV on Xenium slides
            unique_col = next(
                (col for col in self._adata.obs.columns
                if col.lower() in ("unique_cell_id", "unique_cell_index")),
                None
            )
            cell_id_col = unique_col or next(
                (col for col in self._adata.obs.columns if col.lower() == "cell_id"),
                None
            )

            if cell_id_col is None:
                w_status.value = "<span style='color:#e74c3c'>❌ Could not find a cell ID column in adata.obs.</span>"
                w_btn_export.disabled = False
                return

            if cell_id_col not in self._df.columns:
                w_status.value = (f"<span style='color:#e74c3c'>❌ Column '{cell_id_col}' not found in "
                                f"df. Make sure to call reset_index() on the watershed output.</span>")
                w_btn_export.disabled = False
                return

            if self._df[cell_id_col].duplicated().any():
                w_status.value = (f"<span style='color:#e74c3c'>❌ Column '{cell_id_col}' has duplicate "
                                f"values in df — use a globally unique cell ID column instead.</span>")
                w_btn_export.disabled = False
                return

            mapped_sample_id = (
                self._adata.obs[cell_id_col]
                .map(self._df.set_index(cell_id_col)["sample_id"])
            )

            # Guard against a silent merge failure (dtype/formatting mismatch
            # between adata.obs[cell_id_col] and df[cell_id_col], or a df that
            # doesn't actually correspond to this adata). Without this check,
            # a fully-NaN mapped_sample_id sails through to
            # .fillna("unassigned") and every cell gets labelled "unassigned"
            # with no indication that the merge itself failed.
            match_frac = mapped_sample_id.notna().mean()
            if match_frac < 0.5:
                w_status.value = (
                    f"<span style='color:#e74c3c'>❌ Only {match_frac:.0%} of cells in adata.obs "
                    f"matched '{cell_id_col}' values in df. Check that adata and df come from the "
                    f"same cells and that '{cell_id_col}' has matching dtype/formatting in both.</span>"
                )
                w_btn_export.disabled = False
                return

            condition_col = mapped_sample_id.map(self._conditions).fillna("unassigned")
            self._adata.obs["label"] = condition_col

            # Write the STAIA sample id under its namespaced column — but only
            # if that name isn't already taken. If it is, warn instead of
            # overwriting; "label" and the CSV outputs below are unaffected
            # either way, so the export still completes.
            collision_warning = None
            if self.OBS_SAMPLE_ID_COL in self._adata.obs.columns:
                collision_warning = (
                    f"⚠️ adata.obs['{self.OBS_SAMPLE_ID_COL}'] already exists — "
                    f"STAIA left it untouched instead of overwriting it. "
                    f"(adata.obs['label'] and the CSV files were still written normally.)"
                )
            else:
                self._adata.obs[self.OBS_SAMPLE_ID_COL] = mapped_sample_id

            # Save annotated cells from adata.obs. sample_id is included here
            # as a plain output column (freshly computed, independent of
            # whichever branch above ran) so the exported CSV is always
            # correct and self-contained even if OBS_SAMPLE_ID_COL was skipped.
            cells_path = os.path.join(self._output_dir, "cells_annotated.csv")
            self._adata.obs.assign(
                sample_id=mapped_sample_id,
                condition=condition_col,
            ).to_csv(cells_path, index=False)

            # Conditions summary
            cond_df = pd.DataFrame([
                {"sample_id": k, "label": v,
                "n_cells": int((mapped_sample_id == k).sum())}
                for k, v in self._conditions.items()
            ])
            cond_path = os.path.join(self._output_dir, "sample_labels.csv")
            cond_df.to_csv(cond_path, index=False)

            w_btn_export.disabled = False
            status_lines = [f"✅ Saved!<br>- {cells_path}<br>- {cond_path}<br>- adata.obs['label'] updated"]
            if collision_warning:
                status_lines.append(collision_warning)
                w_status.value = (
                    "<span style='color:#2ecc71'>" + status_lines[0] + "</span><br>"
                    "<span style='color:#f39c12'>" + status_lines[1] + "</span>"
                )
            else:
                status_lines.append(f"adata.obs['{self.OBS_SAMPLE_ID_COL}'] updated")
                w_status.value = (
                    "<span style='color:#2ecc71'>" + status_lines[0] + "<br>- " + status_lines[1] + "</span>"
                )

            self._w_table.clear_output(wait=True)
            with self._w_table:
                display(cond_df)

        w_dropdown.observe(on_dropdown_change)
        w_btn_assign.on_click(on_assign)
        w_btn_export.on_click(on_export)

        _refresh_plot()
        _refresh_table() 

        return widgets.VBox([
            widgets.HTML("<b>Label each sample with an experimental label:</b>"),
            widgets.HBox([w_dropdown, w_condition, w_btn_assign]),
            widgets.HBox([w_btn_export, w_status]),
            widgets.HBox([self._w_plot, self._w_table]),  # persistent output widgets
        ])