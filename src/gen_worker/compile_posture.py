"""WHOSE machine this mint is running on — declared by the entry point.

DESIGN-RULINGS §4.30 (Paul, 2026-08-11), verbatim: *"Be nice, especially on
cozy-local — never saturate the user's only machine. Compile parallelism K is
POSTURE-AWARE: aggressive on a dedicated serving pod, gentle/niced/capped on
cozy-local, accepting a slower local mint to stay polite."*

Every clamp in :mod:`gen_worker.aot_compile_pool` is written for a SERVING POD
with a co-resident tenant: ``SERVING_HEADROOM_CPUS`` keeps two cores for an
asyncio beat and an eager forward, ``ENTRY_RSS_RESERVE_BYTES`` keeps host RAM
so a tenant request does not meet the OOM killer. Those are the right terms for
a datacenter box we bought. They are the wrong terms for a desktop, where the
co-resident workload is a human — who outranks the mint, cannot be scheduled
around, and will simply stop using the product if their machine stops
responding.

THE FACT, AND WHY IT IS DECLARED RATHER THAN SNIFFED
----------------------------------------------------
The fact is *"a person is sitting at this machine"*, and no process can measure
it. Three plausible-looking proxies exist in this tree and every one of them is
a DIFFERENT question:

* ``local_cell_store.trust_class() == "untrusted"`` — the HUB's verdict on
  whether this hardware may publish a cell. A rented community-cloud pod is
  untrusted and has no human on it; being polite there would slow work we are
  paying for by the second. Trust is not tenancy.
* ``publisher is None`` at ``local_serve``'s call site — a fact about the SINK.
  ``publish_disarmed`` (a pgw#980 probe) makes an ordinary fleet pod look the
  same, and that pod should compile flat out.
* ``worker_goals`` — what the pod was BOUGHT to do. cozy-local was not bought.

pgw#1127 §2 named the failure mode this codebase keeps hitting: *"two
derivations of one fact"*. So this is not derived at all. It is DECLARED, once,
by the process entry that knows: :mod:`gen_worker.local_serve` — the cozy-local
CLI's only arming entry — passes :data:`USER_MACHINE`, and everything else gets
:data:`FLEET` by construction because that is the struct's default.

And it is deliberately not an environment variable. §1.17, verbatim: *"Envs are
for secrets and configuration, not for logic gates."* Politeness changes what
the machine DOES, so it travels as a typed value on the parent->child
``MintRequest`` — the same wire that already carries ``vram_cap_bytes`` and
``device`` — never as an ambient toggle a stray shell export could flip.

WHAT POLITENESS MEANS, AND WHY EACH TERM
-----------------------------------------
The policy lives here, as methods, so no caller ever branches on the boolean
and the four terms cannot drift apart:

* :meth:`nice_level` — **the primary CPU lever, and the only one that actually
  preserves interactivity.** Reserving cores does not: the scheduler will still
  put a compile on the core the compositor wants. Priority says who yields, at
  microsecond granularity, in the kernel. It is also strictly better than a
  reservation in the other direction — a niced mint uses the WHOLE machine
  while the user is away and gets out of the way the moment they come back, so
  politeness costs nothing when nobody is looking.
* :meth:`cpu_budget_cores` — nice does nothing for the things that are not CPU
  time. K children mean K concurrent ``cc1plus``, K inductor caches being
  written, and K working sets in the page cache; that pressure is felt as
  stutter whatever the priority. ``CPUS_PER_ENTRY_WORKER`` is an AVERAGE (one
  core for ~71 % of an entry, up to ``compile_threads`` for the rest), so a pod
  deliberately overcommits: at the burst it asks for ~2x the box, which is
  correct when the box is ours and otherwise idle. Halving the budget makes the
  BURST ask for the machine instead of double it.
* :meth:`entry_ceiling` — the disk and the page cache being contended are the
  user's own. Half the fleet ceiling.
* :meth:`rss_reserve_bytes` — **a mint that OOMs a desktop is worse than a slow
  one.** ``MemAvailable`` counts reclaimable page cache as free, so a pool sized
  against it will evict the working set of every application the user has open
  before it meets any limit. The reserve is what stops that, and on a desktop
  the OOM killer's most attractive target is a browser with forty tabs.

WHAT IS DELIBERATELY NOT HERE
------------------------------
**No interactivity probe.** "Pause the mint while the user is typing" was
considered and rejected: it is a cross-platform mess (X11 idle time, Wayland
has no equivalent, macOS has a third), and it is wrong in the case that matters
— a developer running a build in a terminal produces no input events at all and
is exactly the person who needs the CPU. :meth:`nice_level` already IS the
yield, implemented by the scheduler, for free.

**No ionice.** It would be the right instinct — priority for I/O the way nice
is for CPU — but ``ioprio_set`` has no stdlib wrapper (a ctypes syscall), and
on the ``none``/``mq-deadline`` schedulers that ship on modern desktops it is a
no-op. The RAM reserve and the entry ceiling bound the I/O instead, by bounding
the number of concurrent writers. Revisit only with a measurement showing disk
contention, not on instinct.

**No third posture.** A community-cloud pod is untrusted hardware that is still
rented by the second with nobody sitting at it; it compiles as FLEET, and that
is not an oversight.
"""

from __future__ import annotations

from typing import Optional

import msgspec

#: The lowest scheduling priority a process may ask for. Not a tunable: there
#: is no case for "somewhat polite" on a machine whose owner is using it, and
#: a niced compile still runs at full speed on an otherwise idle box.
USER_MACHINE_NICE = 19

#: The share of the pod's core budget a mint may take on a user's machine.
#: See :meth:`CompilePosture.cpu_budget_cores` for the derivation.
USER_MACHINE_CPU_SHARE = 2

#: Entry children a user's machine may run at once, against
#: ``aot_compile_pool.MAX_ENTRY_WORKERS`` (8) on a pod. Half, because the disk
#: whose write amplification that constant bounds is the user's own — and
#: because K only buys whole ROUNDS, so the difference between 4 and 8 is one
#: extra round on the largest real family, paid in wall-clock a desktop mint
#: is already trading away.
USER_MACHINE_MAX_ENTRY_WORKERS = 4

#: Host RAM a user's machine keeps for its owner, against
#: ``aot_compile_pool.ENTRY_RSS_RESERVE_BYTES`` (4 GiB) on a pod. Sized as one
#: browser plus one editor plus the page cache they are both living in — the
#: things whose eviction IS the experience of a machine "getting slow", and
#: which ``MemAvailable`` cheerfully reports as free.
USER_MACHINE_RSS_RESERVE_BYTES = 8 * 1024**3


class CompilePosture(msgspec.Struct, frozen=True, kw_only=True):
    """Whose machine a mint runs on. Passed, never inferred.

    Travels on ``mint_process.MintRequest`` so the process that sizes the
    compile pool — the mint child, three frames below the entry that knows — is
    told rather than left to guess.
    """

    #: True when the machine running this mint belongs to the person using the
    #: product, and they are on it. cozy-local, and nothing else today.
    user_machine: bool = False

    # -- the policy, so nothing else branches on the flag ------------------

    def nice_level(self) -> int:
        """Scheduling-priority increment for the mint process tree.

        Applied by the mint child TO ITSELF and inherited by every descendant
        — the entry-pool children, inductor's compile workers, and every
        ``cc1plus`` under them. 0 on a pod: a mint there competes with a tenant
        whose reserves are already held back explicitly, and de-prioritising it
        on top of that would slow paid work for no gain.
        """
        return USER_MACHINE_NICE if self.user_machine else 0

    def cpu_budget_cores(self, vcpus: int, *, headroom: int) -> int:
        """Cores the pool may size itself against.

        A pod takes everything but the tenant's ``headroom``. A user's machine
        takes the NARROWER of that and half the box, so that
        ``K * compile_threads`` at the burst asks for the machine rather than
        double it. The two bounds compose the way a reader expects at both
        ends: on a 4-core laptop the headroom binds (4-2 = 2 -> K=1, serial);
        on a 32-core workstation the half binds (16, not 30).
        """
        budget = int(vcpus) - int(headroom)
        if self.user_machine:
            budget = min(budget, int(vcpus) // USER_MACHINE_CPU_SHARE)
        return budget

    def entry_ceiling(self, default: int) -> int:
        """The hard cap on concurrent entry children. Never widens."""
        if self.user_machine:
            return min(int(default), USER_MACHINE_MAX_ENTRY_WORKERS)
        return int(default)

    def rss_reserve_bytes(self, default: int) -> int:
        """Host RAM the pool must leave alone. Never shrinks."""
        if self.user_machine:
            return max(int(default), USER_MACHINE_RSS_RESERVE_BYTES)
        return int(default)

    def facts(self) -> dict:
        """The posture as it rides the width row, so a K nobody expected can
        be explained without re-deriving anything."""
        return {
            "posture": "user-machine" if self.user_machine else "fleet",
            "nice": self.nice_level(),
        }


#: A pod we bought: no human on it, compile flat out. The default everywhere,
#: which is what keeps the serving path unchanged by construction.
FLEET = CompilePosture(user_machine=False)

#: A machine whose owner is sitting at it. Declared by ``local_serve`` only.
USER_MACHINE = CompilePosture(user_machine=True)


_INSTALLED: Optional[CompilePosture] = None


def install(posture: CompilePosture) -> None:
    """Publish this process's posture.

    One carrier, one moment (§4.22), exactly like ``worker_goals.install``:
    the mint child publishes what its request declared, before anything sizes
    a pool. A test installs a value; it does not clear a cache.
    """
    global _INSTALLED
    _INSTALLED = posture


def current() -> CompilePosture:
    """The installed posture, or :data:`FLEET`.

    The fallback is the right reading for a library import and for every
    serving pod, and it is why nothing on the fleet path had to change.
    """
    return FLEET if _INSTALLED is None else _INSTALLED


__all__ = [
    "USER_MACHINE_MAX_ENTRY_WORKERS",
    "USER_MACHINE_NICE",
    "USER_MACHINE_RSS_RESERVE_BYTES",
    "CompilePosture",
    "FLEET",
    "USER_MACHINE",
    "current",
    "install",
]
