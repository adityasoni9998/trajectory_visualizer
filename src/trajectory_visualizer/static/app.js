"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  step: null,
  file: null,
  version: null,
  groups: [],
  openedGroups: new Set(),
  expanded: new Set(),
  stepRequest: 0,
  detailRequest: 0,
  refreshing: false,
};
const initial = new URLSearchParams(location.hash.slice(1));
const number = (value) => typeof value === "number" && Number.isFinite(value);
const format = (value) => (number(value) ? value.toFixed(3) : "—");
const signed = (value) =>
  number(value) ? `${value > 0 ? "+" : ""}${value.toFixed(3)}` : "—";
const color = (value) =>
  number(value) && value < 0
    ? "negative"
    : number(value) && value > 0
      ? "positive"
      : "";

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = String(text);
  return node;
}

function status(message = "", error = false) {
  $("status").textContent = message;
  $("status").className = error ? "error" : "";
}

async function api(path, params = {}) {
  const response = await fetch(`${path}?${new URLSearchParams(params)}`);
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function saveLocation() {
  const hash = new URLSearchParams({ step: state.step });
  if (state.file) hash.set("file", state.file);
  history.replaceState(null, "", `#${hash}`);
}

function renderSteps(steps) {
  $("step-count").textContent = steps.length;
  $("steps").replaceChildren(
    ...steps.map((step) => {
      const button = el(
        "button",
        `step-button${step === state.step ? " active" : ""}`,
      );
      button.append(el("span", "", "Step"), el("span", "step-number", step));
      button.setAttribute(
        "aria-current",
        step === state.step ? "true" : "false",
      );
      button.addEventListener("click", () => loadStep(step));
      return button;
    }),
  );
}

function renderGroups() {
  const list = $("groups");
  const scroll = list.scrollTop;
  const query = $("search").value.toLowerCase().trim();
  const groups = state.groups.filter(
    (group) =>
      group.task_id.toLowerCase().includes(query) ||
      group.rollouts.some((row) => row.file.toLowerCase().includes(query)),
  );
  if ($("sort").value === "task")
    groups.sort((a, b) => a.task_id.localeCompare(b.task_id));
  $("task-count").textContent =
    `${groups.length} task${groups.length === 1 ? "" : "s"}`;
  const fragment = document.createDocumentFragment();
  for (const group of groups) {
    const container = el("section", "group");
    const button = el("button", "group-button");
    const name = el("div", "group-name", group.task_id);
    name.append(
      el(
        "div",
        "group-meta",
        `${group.rollouts.length} rollouts · ${group.finished} finished`,
      ),
    );
    button.append(
      name,
      el("span", "group-average", `avg ${format(group.average_reward)}`),
    );
    const open = state.openedGroups.has(group.task_id) || Boolean(query);
    button.setAttribute("aria-expanded", String(open));
    button.addEventListener("click", () => {
      if (state.openedGroups.has(group.task_id))
        state.openedGroups.delete(group.task_id);
      else state.openedGroups.add(group.task_id);
      renderGroups();
    });
    container.append(button);
    if (open)
      group.rollouts.forEach((row, index) => {
        if (
          query &&
          !group.task_id.toLowerCase().includes(query) &&
          !row.file.toLowerCase().includes(query)
        )
          return;
        const rollout = el(
          "button",
          `rollout${row.file === state.file ? " active" : ""}`,
        );
        rollout.title = row.file;
        rollout.setAttribute(
          "aria-label",
          `Rollout ${index + 1}, reward ${format(row.reward)}, ${row.file}`,
        );
        rollout.setAttribute(
          "aria-current",
          row.file === state.file ? "true" : "false",
        );
        const label = el(
          "div",
          "",
          `${row.error ? "! " : ""}Rollout ${String(index + 1).padStart(2, "0")}`,
        );
        label.append(
          el(
            "div",
            "rollout-sub",
            `${row.turn_count} steps · ${row.error ? "error" : row.is_finished ? "finished" : "running"}`,
          ),
        );
        rollout.append(
          label,
          el("span", `reward ${color(row.reward)}`, format(row.reward)),
        );
        rollout.addEventListener("click", () => selectRollout(row));
        container.append(rollout);
      });
    fragment.append(container);
  }
  if (!groups.length)
    fragment.append(
      el(
        "p",
        "empty-list",
        state.groups.length
          ? "No tasks match your search."
          : "No rollout files yet. Refresh to discover new files.",
      ),
    );
  list.replaceChildren(fragment);
  list.scrollTop = scroll;
}

function textBlock(value, key) {
  const text = String(value ?? "");
  const block = el("div", "text-block");
  const pre = el("pre");
  const lines = text.split("\n");
  const preview = lines.slice(0, 10).join("\n").slice(0, 700);
  block.append(pre);
  if (preview.length < text.length) {
    const button = el("button", "expand-text");
    const expandedKey = `text:${key}`;
    const update = () => {
      const expanded = state.expanded.has(expandedKey);
      pre.textContent = expanded ? text : `${preview}\n…`;
      button.textContent = expanded
        ? "Show less"
        : `Show all · ${text.length.toLocaleString()} characters`;
      button.setAttribute("aria-expanded", String(expanded));
    };
    button.addEventListener("click", () => {
      if (state.expanded.has(expandedKey)) state.expanded.delete(expandedKey);
      else state.expanded.add(expandedKey);
      update();
    });
    update();
    block.append(button);
  } else pre.textContent = text || "(empty)";
  return block;
}

function disclosure(label, key, build, className = "context-panel") {
  const details = el("details", className);
  details.append(el("summary", "", label));
  let built = false;
  const populate = () => {
    if (!built) {
      details.append(build());
      built = true;
    }
  };
  if (state.expanded.has(key)) {
    details.open = true;
    populate();
  }
  details.addEventListener("toggle", () => {
    if (details.open) {
      state.expanded.add(key);
      populate();
    } else state.expanded.delete(key);
  });
  return details;
}

function jsonTree(value, key) {
  const tree = el("div", "json-tree");
  for (const [name, child] of Object.entries(value)) {
    const childKey = `${key}/${encodeURIComponent(name)}`;
    if (child !== null && typeof child === "object") {
      const shape = Array.isArray(child)
        ? `[${child.length} items]`
        : `{${Object.keys(child).length} fields}`;
      tree.append(
        disclosure(
          `${name} ${shape}`,
          childKey,
          () => jsonTree(child, childKey),
          "json-tree",
        ),
      );
    } else {
      const row = el("div", "json-entry");
      row.append(el("span", "json-key", name));
      if (
        typeof child === "string" &&
        (child.length > 180 || child.includes("\n"))
      )
        row.append(textBlock(child, childKey));
      else row.append(el("span", "json-scalar", JSON.stringify(child)));
      tree.append(row);
    }
  }
  return tree;
}

function rewardInfo(
  info,
  key,
  label = "Reward info",
  className = "reward-info",
) {
  return disclosure(
    `${label} · ${Object.keys(info).length} fields`,
    key,
    () => jsonTree(info, key),
    className,
  );
}

function eventBlock(label, tool, text, key, isError = false, callId = null) {
  const event = el("section", "event");
  const heading = el("div", "event-heading");
  heading.append(
    el(
      "span",
      `event-label ${label === "OBSERVATION" ? "observation" : ""} ${isError ? "error" : ""}`,
      isError ? `${label} · ERROR` : label,
    ),
  );
  heading.append(el("span", "tool-name", tool));
  if (callId) {
    const id = el("span", "call-id", callId);
    id.title = callId;
    heading.append(id);
  }
  event.append(heading, textBlock(text, key));
  return event;
}

function rewardChart(turns) {
  const panel = el("section", "chart-panel");
  const heading = el("div", "chart-heading", "Step rewards");
  heading.append(
    el("span", "", "Recorded reward per step · select a bar to jump"),
  );
  panel.append(heading);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 700 130");
  svg.setAttribute("role", "group");
  svg.setAttribute(
    "aria-label",
    "Recorded step rewards; activate a bar to jump to its step",
  );
  const values = turns.map((turn) => turn.reward).filter(number);
  if (!values.length) {
    panel.append(el("p", "empty-list", "No step rewards recorded."));
    return panel;
  }
  let low = Math.min(0, ...values),
    high = Math.max(0, ...values);
  if (low === high) {
    low = -0.05;
    high = 0.05;
  }
  const y = (value) => 100 - ((value - low) / (high - low)) * 80;
  const add = (tag, attrs, text = null) => {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs))
      element.setAttribute(key, value);
    if (text !== null) element.textContent = text;
    svg.append(element);
    return element;
  };
  add("line", { x1: 48, x2: 690, y1: y(0), y2: y(0), class: "chart-axis" });
  add("text", { x: 0, y: y(high) + 4, class: "chart-label" }, format(high));
  add("text", { x: 0, y: y(low) + 4, class: "chart-label" }, format(low));
  const width = 630 / Math.max(1, turns.length);
  turns.forEach((turn, position) => {
    if (!number(turn.reward)) return;
    const rect = add("rect", {
      x: 52 + position * width,
      y: Math.min(y(turn.reward), y(0)) - (turn.reward === 0 ? 1 : 0),
      width: Math.max(1, width * 0.7),
      height: Math.max(2, Math.abs(y(turn.reward) - y(0))),
      rx: 1,
      class: `bar ${color(turn.reward)}`,
      tabindex: 0,
      role: "button",
      "aria-label": `Step ${turn.index}: reward ${format(turn.reward)}`,
    });
    const title = document.createElementNS(ns, "title");
    title.textContent = `Step ${turn.index}: ${signed(turn.reward)}`;
    rect.append(title);
    const jump = () => {
      const card = $(`turn-${position}`);
      card.scrollIntoView({ block: "start" });
      card.focus({ preventScroll: true });
    };
    rect.addEventListener("click", jump);
    rect.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        jump();
      }
    });
  });
  add(
    "text",
    { x: 52, y: 122, class: "chart-label" },
    `step ${turns[0].index}`,
  );
  add(
    "text",
    { x: 690, y: 122, class: "chart-label", "text-anchor": "end" },
    `step ${turns.at(-1).index}`,
  );
  panel.append(svg);
  return panel;
}

function renderTrajectory(data, keepScroll = false) {
  const scroll = $("main").scrollTop;
  const container = document.createDocumentFragment();
  const header = el("header", "trajectory-header");
  header.append(
    el("div", "breadcrumb", `Training / Step ${state.step} / Rollout`),
    el("h1", "", data.task_id),
    el("p", "filename", data.file),
  );
  const stats = el("div", "headline-stats");
  for (const [label, value, className] of [
    [
      data.is_finished ? "TOTAL REWARD" : "REWARD SO FAR",
      format(data.reward),
      `reward ${color(data.reward)}`,
    ],
    ["RECORDED STEPS", data.turns.length, ""],
    ["ACTIONS", data.turns.reduce((n, turn) => n + turn.actions.length, 0), ""],
  ]) {
    const stat = el("div");
    stat.append(
      el("span", "stat-label", label),
      el("span", `stat-value ${className}`, value),
    );
    stats.append(stat);
  }
  const statusStat = el("div");
  statusStat.append(
    el("span", "stat-label", "STATUS"),
    el(
      "span",
      `pill${data.is_finished ? "" : " running"}`,
      data.error_message
        ? "Finished with error"
        : data.is_finished
          ? "Finished"
          : "Running",
    ),
  );
  stats.append(statusStat);
  header.append(stats);
  container.append(header);
  if (data.task)
    container.append(
      disclosure("Task / initial instruction", "task", () =>
        textBlock(data.task, "task-body"),
      ),
    );
  if (data.turns.length) container.append(rewardChart(data.turns));
  const caption = el("div", "section-caption", "TRAJECTORY");
  caption.append(
    el("span", "", "Reward details collapsed · long text truncated"),
  );
  container.append(caption);
  data.turns.forEach((turn, position) => {
    const key = `turn:${position}`;
    const card = el("article", "turn");
    card.id = `turn-${position}`;
    card.tabIndex = -1;
    const heading = el("header", "turn-header");
    const title = el("div", "turn-title");
    title.append(
      el("span", "turn-index", String(turn.index).padStart(2, "0")),
      el("h3", "", `Step ${turn.index}`),
    );
    const rewards = el("div", "turn-rewards");
    for (const [label, value] of [
      ["Step reward", turn.reward],
      ["Cumulative", turn.cumulative_reward],
    ]) {
      const item = el("span", "", label);
      item.append(el("span", `reward ${color(value)}`, signed(value)));
      rewards.append(item);
    }
    heading.append(title, rewards);
    card.append(heading);
    const body = el("div", "turn-body");
    turn.actions.forEach((action, index) => {
      body.append(
        eventBlock(
          "ACTION",
          action.tool_name,
          action.detail || action.summary,
          `${key}:a:${index}`,
          false,
          action.tool_call_id,
        ),
      );
      action.observations.forEach((observation, j) =>
        body.append(
          eventBlock(
            "OBSERVATION",
            observation.tool_name,
            observation.text,
            `${key}:a:${index}:o:${j}`,
            observation.is_error,
            observation.tool_call_id,
          ),
        ),
      );
    });
    turn.unmatched_observations.forEach((observation, index) =>
      body.append(
        eventBlock(
          "OBSERVATION",
          observation.tool_name,
          observation.text,
          `${key}:o:${index}`,
          observation.is_error,
          observation.tool_call_id,
        ),
      ),
    );
    if (!turn.actions.length && !turn.unmatched_observations.length)
      body.append(
        el(
          "p",
          "empty-list",
          "No action or observation recorded for this step.",
        ),
      );
    if (Object.keys(turn.reward_info).length)
      body.append(rewardInfo(turn.reward_info, `${key}:reward`));
    card.append(body);
    container.append(card);
  });
  if (data.finish_message) {
    const section = el("section", "context-panel");
    section.append(
      el("h3", "section-caption", "FINAL MESSAGE"),
      textBlock(data.finish_message, "final"),
    );
    container.append(section);
  }
  if (data.error_message || data.parse_warnings.length) {
    const notice = el("section", "notice");
    notice.append(
      el("h3", "", data.error_message ? "Trajectory error" : "Parse warnings"),
      textBlock(
        [data.error_message, ...data.parse_warnings].filter(Boolean).join("\n"),
        "errors",
      ),
    );
    container.append(notice);
  }
  if (Object.keys(data.reward_components).length)
    container.append(
      rewardInfo(
        data.reward_components,
        "latest",
        "Last observed numeric diagnostics",
        "context-panel",
      ),
    );
  if (Object.keys(data.terminal_reward_info).length)
    container.append(
      rewardInfo(
        data.terminal_reward_info,
        "terminal",
        "Terminal reward info",
        "context-panel",
      ),
    );
  $("trajectory").replaceChildren(container);
  $("main").scrollTop = keepScroll ? scroll : 0;
}

async function selectRollout(row, keepScroll = false) {
  const request = ++state.detailRequest;
  const step = state.step;
  if (state.file !== row.file) state.expanded.clear();
  state.file = row.file;
  state.version = null;
  renderGroups();
  saveLocation();
  if (!keepScroll)
    $("groups")
      .querySelector(".rollout.active")
      ?.scrollIntoView({ block: "nearest" });
  status("Loading trajectory…");
  try {
    const data = await api("/api/trajectory", { step, file: row.file });
    if (request !== state.detailRequest || state.step !== step) return;
    renderTrajectory(data, keepScroll);
    state.version = row.version;
    status();
  } catch (error) {
    if (request === state.detailRequest) {
      $("trajectory").replaceChildren();
      status(error.message, true);
    }
  }
}

async function loadStep(step, preserve = false, requestedFile = null) {
  const request = ++state.stepRequest;
  if (!preserve || state.step !== step) {
    ++state.detailRequest;
    state.file = null;
    state.version = null;
    state.expanded.clear();
    state.openedGroups.clear();
    state.groups = [];
    renderGroups();
    $("trajectory").replaceChildren(
      el("p", "empty-list", "Choose a rollout to inspect its trajectory."),
    );
  }
  state.step = step;
  for (const button of $("steps").children) {
    const active = button.querySelector(".step-number").textContent === step;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", String(active));
  }
  $("step-title").textContent = `Training step ${step}`;
  status("Indexing rollout files…");
  try {
    const data = await api("/api/rollouts", { step });
    if (request !== state.stepRequest) return;
    state.groups = data.groups;
    $("run-stats").replaceChildren();
    for (const [label, value] of [
      ["rollouts", data.total],
      ["finished", data.finished],
      ["avg reward", format(data.average_reward)],
    ]) {
      const stat = el("span");
      stat.append(
        el("strong", "", value),
        document.createTextNode(` ${label}`),
      );
      $("run-stats").append(stat);
    }
    const rows = data.groups.flatMap((group) => group.rollouts);
    const row =
      rows.find((item) => item.file === (requestedFile || state.file)) ||
      rows[0];
    if (row) {
      state.openedGroups.add(row.task_id);
      renderGroups();
      if (state.file !== row.file || state.version !== row.version)
        await selectRollout(row, preserve && state.file === row.file);
      else status();
    } else {
      state.file = null;
      state.version = null;
      $("trajectory").replaceChildren(
        el("p", "empty-list", "No rollout files in this step yet."),
      );
      renderGroups();
      status();
    }
    saveLocation();
  } catch (error) {
    if (request === state.stepRequest) status(error.message, true);
  }
}

async function refresh(first = false) {
  if (state.refreshing) return;
  state.refreshing = true;
  $("refresh").disabled = true;
  try {
    const data = await api("/api/steps");
    $("root").textContent = data.root;
    $("root").title = data.root;
    const selected = first ? initial.get("step") : state.step;
    const step = data.steps.includes(selected) ? selected : data.steps.at(-1);
    state.step = step;
    renderSteps(data.steps);
    if (step !== undefined)
      await loadStep(step, !first, first ? initial.get("file") : null);
    else {
      $("step-title").textContent = "No training steps yet";
      status(
        "Waiting for numeric step directories containing events/. Use Refresh to check again.",
      );
    }
  } catch (error) {
    status(error.message, true);
  } finally {
    state.refreshing = false;
    $("refresh").disabled = false;
  }
}

$("refresh").addEventListener("click", () => refresh());
$("search").addEventListener("input", renderGroups);
$("sort").addEventListener("change", renderGroups);
$("live").addEventListener("change", () => {
  if ($("live").checked) refresh();
});
setInterval(() => {
  if ($("live").checked && !document.hidden) refresh();
}, 30000);
refresh(true);
