"""agents.py - AIs that work in the household as members: a profile per AI (library kind "agent"), what it has been granted, and how it hands work to sub-agents within that grant.
An AI is treated like a person who was given access: it reads what it was let see, keeps its own notes without asking, and proposes every change to anything that is not its own. Safety and trust come from the stakes rules (proposals, masks, review), the same for any model behind it - a small local one, a large one, an outside one.

A profile: {"name", "about", "owner": the portal user who grants it (its work runs for them, and it never has more than they have), "users": [others who may work with it],
  "prompts" / "contexts": [library ids]: instruction blocks kept in the library, put before / after "system": standing instructions written by a person (the AI only proposes changes to them),
  how its model runs: "num_ctx", "rounds", "kv_cache_type", "reply_tokens", "think", "temperature" (a run's own values stand above them),
  what it may reach, as fields (library kind agent) or under "grants" (older profiles; a field stands above the same grant there):
  "read_first": [{"project", "file"}]: files of a shared project put before it at the start of every run (preferences, skills, house rules - shared with the person, changed by proposal),
  "grants": {
    "space": its own space (agent_spaces/<name>): notes, plans, what it thinks matters - written directly, it is the only stakeholder,
    "projects": true or [slugs]: Muninn projects worked through their masks as agent:<profile id> (state, issues, plans; what its mask lets it write directly, the rest proposed),
    "files": a library space it reads and changes by proposal only (the code), with the code map,
    "tools": [pipeline ids]: each a tool (knowledge, web, measurement ...),
    "skills": [skill ids]: their texts join its instructions,
    "delegate": {"most": sub-tasks per run, "rounds": per sub-task}: it may hand a task to a sub-agent; absent: it may not,
    "background": {"most": tasks going at once}: it may start its tools as runs of their own that it does not wait for, check on and stop them; absent: it may not}}
A run names the profile (agent step config "profile"); what the run asks for beyond the grant is refused, what it asks for within it narrows it. A sub-agent works as the same AI (the same masks) with a subset of its grants and never more.
The agent loop reads profiles through PREPARE and serves the toolset "agent" (what it has, delegate)."""
import json, time, asyncio
from . import agent_loop as AL, steps

def _may_use(pr: dict, who: str) -> bool: return not pr.get("owner") or who in (pr["owner"], *(pr.get("users") or [])) or "*" in (pr.get("users") or [])   # works for no one person, or for this one, or for everyone ("*")

GRANTS = ("space", "projects", "files", "tools", "skills", "delegate", "background")
RUNS = {"num_ctx": "num_ctx", "rounds": "rounds"}   # profile field -> agent config key, used where the run sets none
OPTIONS = {"kv_cache_type": "kv_cache_type", "reply_tokens": "num_predict", "temperature": "temperature"}   # profile field -> request option, under what the run asks itself

def grants(pr: dict) -> dict:
    """What a profile grants: its fields (library kind agent), over its older "grants" object. delegate_most 0 means it may not hand work on."""
    g = {**(pr.get("grants") or {}), **{k: pr[k] for k in ("space", "projects", "files", "tools", "skills") if pr.get(k) not in (None, "", [])}}
    if "delegate_most" in pr: g = {k: v for k, v in g.items() if k != "delegate"} | ({"delegate": {"most": int(pr["delegate_most"]), "rounds": int(pr.get("delegate_rounds") or 30)}} if int(pr.get("delegate_most") or 0) > 0 else {})
    if "background_most" in pr: g = {k: v for k, v in g.items() if k != "background"} | ({"background": {"most": int(pr["background_most"])}} if int(pr.get("background_most") or 0) > 0 else {})
    return g

def _lib(ctx): return ctx.director.aviary.lib

def _within(name: str, asked, granted):
    """What a run may have of one grant: the grant itself when it asks nothing, else what it asks if the grant covers it; PermissionError otherwise."""
    if asked in (None, "", [], False): return granted
    if name == "projects" and granted is True: return asked
    if isinstance(granted, list) and isinstance(asked, list) and set(map(str, asked)) <= set(map(str, granted)): return asked
    if asked == granted: return asked
    raise PermissionError(f"this AI is not granted {name} {asked!r} (granted: {granted!r})")

def prepare(cfg: dict, ctx) -> dict:
    """An agent step's config with its profile applied: the grants (narrowed by what the config asks), the identity agent:<id> for project masks, its standing instructions and skills, and its sub-agent rules. A step with no profile is left as it is."""
    if "profile" not in cfg: return cfg
    owner = str(ctx.run.meta.get("owner") or "")
    if not cfg["profile"]:   # a pipeline that takes a profile but was given none: the one AI this person may work with, else ask
        mine = [x["id"] for x in _lib(ctx).list("agent") if _may_use(x, owner)]
        if len(mine) != 1: raise ValueError(f"""choose which AI works here (its profile){": " + ", ".join(mine) if mine else ": there is none yet (library kind agent)"}""")
        cfg = {**cfg, "profile": mine[0]}
    aid = str(cfg["profile"])
    pr = _lib(ctx).get("agent", aid)
    if not _may_use(pr, owner): raise PermissionError(f"""the AI {aid} works for {pr["owner"]}{" and " + ", ".join(pr["users"]) if pr.get("users") else ""}, not for {owner or "work without an owner"}""")
    g = grants(pr)
    out = {**cfg, **{k: _within(k, cfg.get(k), g.get(k)) for k in GRANTS if k in g or cfg.get(k)}, "as": f"agent:{aid}", "author": cfg.get("author") or f"agent:{aid}"}
    out.update({to: pr[k] for k, to in RUNS.items() if pr.get(k) and not cfg.get(to)})
    out.update({k: [*(pr.get(k) or []), *[x for x in cfg.get(k) or [] if x not in (pr.get(k) or [])]] for k in ("prompts", "contexts") if pr.get(k) or cfg.get(k)})
    body, opts = cfg.get("body") or {}, {to: pr[k] for k, to in OPTIONS.items() if pr.get(k) not in (None, "", 0)}
    if opts or pr.get("think"): out["body"] = {**({"think": pr["think"] == "on"} if pr.get("think") else {}), **body, "options": {**opts, **(body.get("options") or {})}}
    skills = "\n\n".join(str(_lib(ctx).get("skill", s).get("text", "")) for s in out.get("skills") or [] if _lib(ctx).has("skill", s))
    out["system"] = "\n\n".join(x for x in (str(pr.get("system") or ""), str(cfg.get("system") or ""), f"Skills:\n{skills}" if skills else "") if x)
    out["_profile"] = {"id": aid, **pr}
    return out

class AgentTools:
    """What the AI has, said at the start and on asking (my_grants); delegate when it may hand work on. Served when the step names a profile."""
    def __init__(self, cfg, ctx): self.cfg, self.ctx, self.pr, self.given, self.binding = cfg, ctx, {**cfg["_profile"], "grants": grants(cfg["_profile"])}, 0, asyncio.Lock()   # binding: sub-agents asked together place the helper role once
    def _have(self) -> dict: return {"you": self.cfg["as"], "name": self.pr.get("name") or self.pr["id"], "working for": self.ctx.run.meta.get("owner") or "nobody", **{k: self.cfg.get(k) for k in GRANTS if self.cfg.get(k)}}
    async def open(self) -> str:
        first = "\n\n".join(self._read_first())
        d = self.cfg.get("delegate")
        return f"""You are {self.pr.get("name") or self.pr["id"]} ({self.cfg["as"]}), a member of this household's system working for {self.ctx.run.meta.get("owner") or "nobody"}. What you have (my_grants repeats it): {json.dumps(self._have(), ensure_ascii=False)}.
How changes work here: your own space is yours (write freely: notes, plans, what you think matters). Everything else you change by proposal - project files beyond what your mask lets you write, code files, shared preferences and skills - and a person accepts or declines each one. Say why in each proposal. Nothing is lost: earlier versions are kept.{f" You may hand a task to a sub-agent (delegate, at most {d.get('most', 3)} per run): it works as you, with the part of your grants you give it, and its proposals wait for review like yours." if d else ""}{" You may start your tools as tasks that go on without you (start_task), check on them in a later answer (task_status) and stop them; say to the person which tasks you started and what each is for." if self.cfg.get("background") else ""}""" + (f"\n\nShared with the person (read at every start; propose changes to them like any file):\n{first}" if first else "")
    def _read_first(self) -> list:
        rf = self.pr.get("read_first") or []
        if not rf or not self.cfg.get("projects"): return []
        from tools.muninn import projects as PJ   # Muninn's projects: present when the projects grant is (it serves that toolset)
        who = (self.cfg["as"], str(self.ctx.run.meta.get("owner") or ""))
        return [f"""--- {x["project"]}/{x["file"]} ---\n{PJ.read(x["project"], x["file"], who)[:12000]}""" for x in rf]
    def _mine(self) -> list: return [r for r in self.ctx.director.runs.values() if r.meta.get("by") == self.cfg["as"] and r.meta.get("owner") == self.ctx.run.meta.get("owner")]   # tasks this AI started for this person, from any of its runs

    def _task(self, r) -> dict:
        """A task as the AI reads it: its state, what is working, why it stopped, and what it gave once done (each value cut to what fits a tool result)."""
        v, gives = r.view(), [g for s in r.plan["steps"][-1:] for g in s.get("gives", [])]
        return {"task": r.id, "label": r.meta.get("label", ""), "tool": r.meta.get("tool", ""), "state": r.meta["state"], "for_s": round(time.time() - r.meta.get("created", time.time())), "working": v["busy"], **({"error": r.meta["error"]} if r.meta.get("error") else {}), **({"results": {k: str(r.data.get(k))[:4000] for k in gives if k in r.data}} if r.meta["state"] == "done" else {})}

    async def _background(self, name, a):
        """start_task / task_status / stop_task: its tools as runs of their own (Aviary.beside), at most the grant's most going at once."""
        if name == "task_status":
            rs = [r for r in self._mine() if not a.get("task") or r.id == a["task"]]
            return bool(rs) or not a.get("task"), json.dumps([self._task(r) for r in sorted(rs, key=lambda r: r.meta.get("created", 0))[-20:]] or ("no such task of yours" if a.get("task") else "no tasks started"), ensure_ascii=False, default=str)
        if name == "stop_task":
            r = next((r for r in self._mine() if r.id == a.get("task")), None)
            if not r: return False, f"""no task of yours named {a.get("task")!r} (task_status lists them)"""
            self.ctx.director.stop(r.id, now=bool(a.get("now")))
            return True, json.dumps(self._task(r), ensure_ascii=False, default=str)
        going = [r for r in self._mine() if r.meta["state"] in ("new", "running", "stopping", "held")]
        if len(going) >= int(self.cfg["background"].get("most", 1)): return False, f"""{len(going)} of your tasks are going, your most: wait for one (task_status) or stop one"""
        tools = {t["name"]: t for t in (steps._tool(self.ctx.director.aviary, x) for x in self.cfg.get("tools") or [])}
        if a.get("tool") not in tools: return False, f"""start_task runs one of your tools: {", ".join(tools) or "you have none"} - not {a.get("tool")!r}"""
        t, me = tools[a["tool"]], self.ctx.run
        rid = f"""task-{self.pr["id"]}-{int(time.time() * 1000) % 10**10}"""
        st = await self.ctx.director.aviary.beside(me, rid, t["plan"], {**{k: v for k, v in (a.get("arguments") or {}).items() if k in (t["plan"].get("params") or {})}, **t["fixed"]}, self.cfg.get("roles"), label=str(a.get("label") or a["tool"]))
        self.ctx.director.runs[rid].meta.update(by=self.cfg["as"], tool=a["tool"])
        return st not in ("failed", "unplaced"), json.dumps(self._task(self.ctx.director.runs[rid]), ensure_ascii=False, default=str)

    async def tools(self) -> list: return [AL._tool("my_grants", "What you have been granted: your own space, projects, the code files, tools, skills, whether you may delegate.", {}, read_only=True), *([AL._tool("delegate", "Hand one task to a sub-agent and wait for its result; several delegate calls in one reply run at the same time (give each a separate part: different files or questions). It works as you with only what you give it here, and finishes with a summary; its changes are proposals like yours. Give it a task that stands on its own: what to change, where, how to check it.", {"task": AL._p("string", "the whole task, standing on its own"), "tools": {"type": "array", "items": {"type": "string"}, "description": "pipelines it may use: a subset of yours (empty: none)"}, "files": AL._p("boolean", "whether it reads and proposes to the code files"), "projects": {"type": "array", "items": {"type": "string"}, "description": "projects it may work on: a subset of yours (empty: none)"}, "helper": AL._p("boolean", "true: the smaller helper model does it (simple, well-bounded work)")}, ["task"], together=True)] if self.cfg.get("delegate") else []), *([AL._tool("start_task", "Start one of your tools as a task of its own and do not wait for it: it goes on after your answer (it is in the person's runs list). Use it for slow work (pictures, long research) or several things at once; check it later with task_status and take its results into the next step yourself.", {"tool": AL._p("string", "which of your tools"), "arguments": {"type": "object", "description": "the tool's arguments, as you would call it"}, "label": AL._p("string", "a few words saying what this task is for, shown to the person")}, ["tool", "arguments"]), AL._tool("task_status", "Your tasks: each one's state, what is working, why it stopped, and its results once done. Name one, or none for all.", {"task": AL._p("string", "a task's id (start_task answered it); empty: all of yours")}, [], read_only=True), AL._tool("stop_task", "Stop one of your tasks: it finishes the step it is on, or at once with now.", {"task": AL._p("string", "its id"), "now": AL._p("boolean", "stop at once; work under way is lost")}, ["task"])] if self.cfg.get("background") else [])]
    async def call(self, name, a):
        if name == "my_grants": return True, json.dumps(self._have(), ensure_ascii=False)
        if name in ("start_task", "task_status", "stop_task"): return await self._background(name, a)
        d = self.cfg["delegate"]
        if self.given >= int(d.get("most", 3)): return False, f"""this run has handed on {self.given} tasks, its most ({d.get("most", 3)}): do the rest yourself or finish"""
        self.given += 1
        n = self.given   # taken before anything waits: sub-agents asked together each get their own number and count against the most
        sub = {"role": "main", "task": str(a["task"]), "mode": "task", "as": self.cfg["as"], "author": self.cfg.get("author"), "rounds": int(d.get("rounds", 30)), "space": f"""{self.cfg["space"]}-work""" if self.cfg.get("space") else "",
               "tools": _within("tools", list(a.get("tools") or []), self.cfg.get("tools") or []) if a.get("tools") else [], "projects": _within("projects", list(a.get("projects") or []), self.cfg.get("projects")) if a.get("projects") else False,
               "files": self.cfg.get("files") if a.get("files") else "", "system": self.cfg.get("system", ""), "give": {"summary": "text", "status": "status", "proposed": "proposed"}}
        if sub["files"]: sub["check"] = {"tool": "check_files"}
        me, sid = self.ctx.run, self.ctx.step["id"]
        role = "helper" if a.get("helper") and "helper" in (me.plan.get("roles") or {}) else "main"
        async with self.binding:
            if role not in (me.meta.get("bound") or {}): await self.ctx.director.aviary.bind(me, {role})
        if not me.meta["bound"].get(role): role = "main"   # no helper model is set or free: the main model does it
        plan = {"name": "Delegated task", "roles": {"main": {"needs": ["chat"], "about": "the sub-agent's model"}}, "params": {}, "steps": [{"id": "work", "type": "agent", "needs": [], "gives": ["summary", "status", "proposed"], "config": sub}]}
        rid = f"{me.id}~{sid}~d{n}"
        st = await self.ctx.director.lend(me, self.ctx.step, self.ctx.director.aviary.sub(me, rid, plan, {}, {"main": role}))   # the model this answer holds is lent to the sub-agent while it waits
        r = self.ctx.director.runs[rid]
        return st == "done", json.dumps({"status": r.data.get("status", st), "summary": r.data.get("summary") or r.meta.get("error", ""), "proposed": r.data.get("proposed"), "run": rid}, ensure_ascii=False, default=str)
    async def close(self, status) -> dict: return {"delegated": self.given}

AL.PREPARE.append(prepare)
AL.register_toolset("agent", AgentTools, lambda cfg: cfg.get("_profile"))
