from __future__ import annotations

"""
Debug interpreter for step-by-step execution of language model programs.

This module provides `LanguageModel_ForwardInterpreter_Debug`, a drop-in
replacement/subclass of `LanguageModel_ForwardInterpreter` that adds an
interactive console debugger with commands similar to a traditional debugger:

- step [n]: Step into the next n instructions (default 1)
- next [n]: Step over within the current scope n instructions (default 1)
- continue: Run without stopping until the current function returns
- finish: Run until the current function returns (stop at caller)
- list [k]: Show the next k instructions in the current scope (default 5)
- ctx: Show current variables with types and brief values
- p <var>: Print a variable value by name from the current scope
- stack: Show the call stack (function and scope hierarchy)
- where: Show the scope hierarchy (path of scopes)
- help: Show available commands

Notes on stepping semantics:
- "step" breaks before the next instruction, descending into nested blocks/calls.
- "next" counts only instructions at the current scope depth (step-over).
- "continue" suppresses all stops until we return from the current function.
- "finish" runs until we return from the current function, then stops.

The implementation carefully hooks the instruction dispatch points while keeping
the core interpreter logic intact to minimize the risk of behavior drift.
"""

from dataclasses import dataclass
from typing import Any, Optional, Iterator

import numpy as np
from numpy.typing import NDArray

import core.lib_impl.lib_classes as lc
from core.lib_impl.lib_interpretor import (
    LanguageModel_ForwardInterpreter,
    ExecutionContext,
)


# --------------------------------------------------------------------------------------
# Debugger state structures
# --------------------------------------------------------------------------------------


@dataclass
class DebugFrame:
    """
    A single frame representing the execution of a block function.
    Tracks the current instruction index within the function's flow control list.
    """

    block_function: lc.BlockFunction
    context: ExecutionContext
    instructions: list[lc.FlowControlInstruction]
    index: int

    def scope_path(self) -> list[str]:
        return [scope.name for scope in self.context.scope_stack]


class DebuggerState:
    """
    Holds the debugger's runtime state and stepping configuration.
    """

    def __init__(self) -> None:
        # Current stepping mode: "pause" | "step" | "next" | "continue" | "finish"
        self.mode: str = "pause"

        # For step/next, how many instructions to execute before pausing again
        self.steps_remaining: int = 0

        # For next: the recursion depth (of instruction dispatch) at which we count steps
        self.step_over_base_depth: int = 0

        # For finish/continue: target call-stack depth at which to stop again
        self.finish_target_call_depth: Optional[int] = None

        # Recursion depth of instruction dispatch; used to implement step-over semantics
        self.recursion_depth: int = 0

        # Stack of DebugFrame for block function calls
        self.call_stack: list[DebugFrame] = []

    # ----------------------------- stack mgmt ----------------------------------
    def push_frame(self, frame: DebugFrame) -> None:
        self.call_stack.append(frame)

    def pop_frame(self) -> None:
        if self.call_stack:
            self.call_stack.pop()

    @property
    def call_depth(self) -> int:
        return len(self.call_stack)

    # ----------------------------- stepping API --------------------------------
    def set_pause(self) -> None:
        self.mode = "pause"
        self.steps_remaining = 0
        self.finish_target_call_depth = None

    def set_step(self, n: int = 1) -> None:
        # Execute n instructions across any depth, then pause
        self.mode = "step"
        self.steps_remaining = max(1, n)
        self.finish_target_call_depth = None

    def set_next(self, n: int = 1) -> None:
        # Execute n instructions in the current scope (depth), stepping over deeper calls
        self.mode = "next"
        self.steps_remaining = max(1, n)
        self.step_over_base_depth = self.recursion_depth
        self.finish_target_call_depth = None

    def set_continue(self) -> None:
        # Run without stopping
        self.mode = "continue"
        self.steps_remaining = 0
        self.finish_target_call_depth = None

    def set_finish(self) -> None:
        # Run until the current function returns, then pause
        self.mode = "finish"
        self.steps_remaining = 0
        self.finish_target_call_depth = self.call_depth - 1 if self.call_depth > 0 else 0


# --------------------------------------------------------------------------------------
# Utility helpers for UI/formatting
# --------------------------------------------------------------------------------------


def _summarize_value(value: Any) -> str:
    """
    Return a brief, helpful string for a value (shape/dtype for arrays, length for lists, etc.).
    """
    try:
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        if isinstance(value, dict):
            return f"dict(len={len(value)})"
        return repr(value)
    except Exception:
        return "<unprintable>"


def _format_instruction(instruction: lc.FlowControlInstruction) -> str:
    try:
        return str(instruction)
    except Exception:
        return f"{instruction.__class__.__name__}()"


class _Ansi:
    """Minimal ANSI color/style helpers for a readable console UI."""

    RESET = "\033[m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    FG_CYAN = "\033[36m"
    FG_MAGENTA = "\033[35m"
    FG_YELLOW = "\033[33m"
    FG_GREEN = "\033[32m"
    FG_BLUE = "\033[34m"
    FG_RED = "\033[31m"

    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


def _c(style: str, text: str) -> str:
    return f"{style}{text}{_Ansi.RESET}"


def _print_scope_path(
    context: ExecutionContext,
    main_block_name: Optional[str] = None,
    current_block_name: Optional[str] = None,
) -> None:
    parts: list[str] = []
    for scope in context.scope_stack:
        name = scope.name
        # Annotate forward_pass with main block name when available
        if name == "forward_pass" and main_block_name:
            name = f"{name}" + _c(_Ansi.DIM, f"[main={main_block_name}]")
        # Annotate function_ scopes with current block name when available
        if name.startswith("function_") and current_block_name:
            name = f"{name}" + _c(_Ansi.DIM, f"[block={current_block_name}]")
        parts.append(_c(_Ansi.FG_MAGENTA, name))
    print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, "Scopes:") + f" {' > '.join(parts)}")


def _find_defining_scope(context: ExecutionContext, var_name: str) -> tuple[Optional[str], Optional[int]]:
    """Return the name and level index for the scope defining var_name, if any."""
    # scope_stack is ordered from global (0) to current (last)
    for idx, scope in enumerate(context.scope_stack):
        if var_name in scope.symbols:
            return scope.name, idx
    return None, None


def _print_context_variables(
    context: ExecutionContext,
    main_block_name: Optional[str] = None,
    current_block_name: Optional[str] = None,
) -> None:
    # Header with scope position
    current_scope = context.current_scope.name
    current_level = len(context.scope_stack) - 1
    print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, "Context:"), end=" ")
    print(
        _c(
            _Ansi.FG_GREEN,
            f"current-scope={current_scope} (level={current_level})",
        )
    )
    _print_scope_path(context, main_block_name, current_block_name)

    # Variables
    print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, "Variables (accessible in current scope):"))
    variables = context.variables
    var_types = context.variable_types
    for name in sorted(variables.keys()):
        v = variables[name]
        vt = var_types.get(name)
        type_str = vt.type_name if hasattr(vt, "type_name") else repr(vt)
        scope_name, scope_idx = _find_defining_scope(context, name)
        scope_info = (
            f" [scope={scope_name}, level={scope_idx}]" if scope_name is not None else ""
        )
        print(
            f"  - "
            f"{_c(_Ansi.BOLD, name)}: "
            f"{_c(_Ansi.DIM, str(type_str))} = {_summarize_value(v)}"
            f"{_c(_Ansi.DIM, scope_info)}"
        )


def _print_call_stack(state: DebuggerState) -> None:
    if not state.call_stack:
        print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, "Call stack:"), _c(_Ansi.DIM, "<empty>"))
        return
    print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, "Call stack (oldest to newest):"))
    for i, frame in enumerate(state.call_stack):
        fn = frame.block_function.function_name
        scope = " > ".join(frame.scope_path())
        pc = f"pc={frame.index}/{len(frame.instructions)}"
        print(
            f"  [{_c(_Ansi.FG_YELLOW, str(i))}] "
            f"{_c(_Ansi.FG_GREEN + _Ansi.BOLD, fn)}  @ "
            f"{_c(_Ansi.FG_MAGENTA, scope)}  "
            f"({_c(_Ansi.DIM, pc)})"
        )


def _print_next_instructions(frame: DebugFrame, k: int = 5) -> None:
    start = frame.index
    end = min(len(frame.instructions), start + max(1, k))
    if start >= len(frame.instructions):
        print(_c(_Ansi.DIM, "No further instructions in this scope."))
        return
    hdr = f"Next instructions in scope [{start}:{end}]:"
    print(_c(_Ansi.FG_CYAN + _Ansi.BOLD, hdr))
    for idx in range(start, end):
        is_next = idx == start
        prefix = _c(_Ansi.FG_YELLOW + _Ansi.BOLD, "->") if is_next else "  "
        idx_str = _c(_Ansi.FG_YELLOW, str(idx))
        instr_str = _format_instruction(frame.instructions[idx])
        instr_str = _c(_Ansi.FG_BLUE if is_next else _Ansi.DIM, instr_str)
        print(f"  {prefix} [{idx_str}] {instr_str}")


def _read_command(prompt: str = "dbg> ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        # Gracefully fallback to continue on input interruptions
        print("\n[debugger] Input interrupted, continuing execution.")
        return "continue"


def _print_help() -> None:
    print(
        _c(_Ansi.FG_CYAN + _Ansi.BOLD, "Commands:")
        + "\n  "
        + _c(_Ansi.FG_GREEN, "step [n]")
        + "       : Step into next instruction(s).\n  "
        + _c(_Ansi.FG_GREEN, "next [n]")
        + "       : Step over in current scope (n instructions).\n  "
        + _c(_Ansi.FG_GREEN, "continue")
        + "       : Run without stopping.\n  "
        + _c(_Ansi.FG_GREEN, "finish")
        + "         : Run until current function returns, then pause.\n  "
        + _c(_Ansi.FG_GREEN, "list [k]")
        + "       : Show next k instructions in this scope.\n  "
        + _c(_Ansi.FG_GREEN, "ctx")
        + "            : Show variables in current scope.\n  "
        + _c(_Ansi.FG_GREEN, "p <var>")
        + "        : Print a variable value.\n  "
        + _c(_Ansi.FG_GREEN, "stack")
        + "          : Show call stack.\n  "
        + _c(_Ansi.FG_GREEN, "where")
        + "          : Show scope hierarchy.\n  "
        + _c(_Ansi.FG_GREEN, "help")
        + "           : Show this help.\n"
    )


# --------------------------------------------------------------------------------------
# Debug interpreter
# --------------------------------------------------------------------------------------


class LanguageModel_ForwardInterpreter_Debug(LanguageModel_ForwardInterpreter):
    """
    Debug-capable forward interpreter.

    It subclasses the standard interpreter and augments the execution with a
    minimal interactive debugger. The core algorithm is intentionally kept as
    close as possible to the base class to avoid divergences.
    """

    def __init__(self, language_model: lc.Language_Model) -> None:
        super().__init__(language_model)
        self._dbg: DebuggerState = DebuggerState()

    # --------------------------- public control API ----------------------------
    def debugger_continue(self) -> None:
        self._dbg.set_continue()

    def debugger_step(self, n: int = 1) -> None:
        self._dbg.set_step(n)

    def debugger_next(self, n: int = 1) -> None:
        self._dbg.set_next(n)

    def debugger_finish(self) -> None:
        self._dbg.set_finish()

    # ------------------------------ core hooks --------------------------------
    def _pause_if_needed(self, frame: DebugFrame, instruction: lc.FlowControlInstruction) -> None:
        """
        Pause execution before the given instruction depending on the stepping mode.
        """
        # Implement stepping decisions
        if self._dbg.mode == "continue":
            return

        if self._dbg.mode == "finish":
            # Don't pause until we return to the caller
            if (
                self._dbg.finish_target_call_depth is not None
                and self._dbg.call_depth <= self._dbg.finish_target_call_depth
            ):
                # We've returned to caller; drop into pause mode now
                self._dbg.set_pause()
            else:
                return

        if self._dbg.mode == "step":
            if self._dbg.steps_remaining > 0:
                # Consume one instruction and keep running
                self._dbg.steps_remaining -= 1
                return
            # No steps remaining: pause
        elif self._dbg.mode == "next":
            # Only count instructions executed at the same recursion depth
            if self._dbg.recursion_depth > self._dbg.step_over_base_depth:
                return
            if self._dbg.steps_remaining > 0:
                self._dbg.steps_remaining -= 1
                return
            # No steps remaining at this depth: pause

        # Default (or mode == "pause"): show UI
        self._interactive_prompt(frame, instruction)

    def _interactive_prompt(self, frame: DebugFrame, instruction: lc.FlowControlInstruction) -> None:
        """
        Simple console REPL to control execution.
        """
        print("\n" + _c(_Ansi.FG_CYAN + _Ansi.BOLD, "[debugger] About to execute instruction:"))
        print("  " + _c(_Ansi.FG_BLUE, _format_instruction(instruction)))
        main_block_name: Optional[str] = None
        if self.language_model.model_blocks:
            if self.language_model.main_block and self.language_model.main_block in self.language_model.model_blocks:
                main_block_name = self.language_model.main_block
            else:
                main_block_name = list(self.language_model.model_blocks.keys())[0]
        current_block_name = None
        try:
            current_block_name = frame.block_function.model_block.block_name
        except Exception:
            pass
        _print_scope_path(frame.context, main_block_name, current_block_name)
        _print_next_instructions(frame, k=5)

        while True:
            cmdline = _read_command("dbg> ")
            if not cmdline:
                continue

            parts = cmdline.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ("help", "h", "?"):
                _print_help()
                continue

            if cmd in ("step", "s"):
                n = int(arg) if arg and arg.isdigit() else 1
                self._dbg.set_step(n)
                break

            if cmd in ("next", "n", "ni"):
                n = int(arg) if arg and arg.isdigit() else 1
                self._dbg.set_next(n)
                break

            if cmd in ("continue", "c"):
                self._dbg.set_continue()
                break

            if cmd in ("finish", "f"):
                self._dbg.set_finish()
                break

            if cmd in ("list", "l"):
                k = int(arg) if arg and arg.isdigit() else 5
                _print_next_instructions(frame, k=k)
                continue

            if cmd == "ctx":
                _print_context_variables(frame.context, main_block_name, current_block_name)
                continue

            if cmd in ("where", "w"):
                _print_scope_path(frame.context, main_block_name, current_block_name)
                continue

            if cmd == "stack":
                _print_call_stack(self._dbg)
                continue

            if cmd == "p" and arg:
                name = arg
                try:
                    value = frame.context.get_variable(name)
                    print(f"{name} = {_summarize_value(value)}")
                except KeyError as e:
                    print(str(e))
                continue

            print("Unknown command. Type 'help' for available commands.")

    # ------------------------------ overrides ---------------------------------
    def _execute_block_function(
        self,
        block_function: lc.BlockFunction,
        context: ExecutionContext,
        function_args: Optional[dict[str, Any]] = None,
    ) -> dict[str, NDArray[np.float32]]:
        """
        Execute a block function with debugger hooks.
        This mirrors the base implementation closely.
        """
        # Create local scope for function execution
        local_context = context.enter_scope(f"function_{block_function.function_name}")

        # Add 'self' to the context; provide access to layer instances
        class SelfWrapper:
            def __init__(self, model_block: lc.ModelBlock, context: ExecutionContext) -> None:
                self.model_block: lc.ModelBlock = model_block
                self.context: ExecutionContext = context

            def __getattr__(self, name: str) -> Any:
                if not hasattr(self.__class__, '_getattr_depth'):
                    self.__class__._getattr_depth = 0
                if self.__class__._getattr_depth > 10:
                    print(f"DEBUG: RecursionError in SelfWrapper.__getattr__ for {name}")
                    return None
                self.__class__._getattr_depth += 1
                try:
                    if self.context.has_variable(name):
                        return self.context.get_variable(name)
                except RecursionError:
                    print(f"DEBUG: RecursionError in SelfWrapper.__getattr__ for {name}")
                    return None
                finally:
                    self.__class__._getattr_depth -= 1
                if hasattr(self.model_block, name):
                    attr: Any = getattr(self.model_block, name)
                    if hasattr(attr, 'block_layers') and hasattr(attr, 'block_name') and 'BlockModuleList' in attr.block_name:
                        layer_instances: list[lc.Layer] = []
                        for layer_name in sorted(attr.block_layers.keys()):
                            if self.context.has_variable(layer_name):
                                layer_instances.append(self.context.get_variable(layer_name))
                        return layer_instances
                    return attr
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            def __iter__(self) -> Iterator[Any]:
                return iter(self.model_block)

        local_context.set_variable("self", lc.VarType("ModelBlock"), SelfWrapper(block_function.model_block, local_context))

        # Set function arguments
        if function_args:
            for arg_name, arg_value in function_args.items():
                if arg_name in block_function.function_arguments:
                    arg_type, _ = block_function.function_arguments[arg_name]
                    local_context.set_variable(arg_name, arg_type, arg_value)

        # Default values for missing arguments
        for arg_name, (arg_type, default_expr) in block_function.function_arguments.items():
            if not local_context.has_variable(arg_name):
                if not isinstance(default_expr, lc.ExpressionNoDefaultArguments):
                    default_value: Any = self._evaluate_expression(default_expr, local_context)
                    local_context.set_variable(arg_name, arg_type, default_value)

        # Prepare debugger frame
        frame = DebugFrame(
            block_function=block_function,
            context=local_context,
            instructions=list(block_function.function_flow_control),
            index=0,
        )
        self._dbg.push_frame(frame)

        outputs: dict[str, NDArray[np.float32]] = {}

        try:
            # Instruction loop with debugger hook
            while frame.index < len(frame.instructions):
                instruction = frame.instructions[frame.index]

                # Pause before executing this instruction if needed
                self._pause_if_needed(frame, instruction)

                # Execute instruction
                _result: Any = self._execute_flow_control_instruction(instruction, local_context)

                # Handle return
                if isinstance(instruction, lc.FlowControlReturn):
                    for return_var in instruction.return_variables:
                        outputs[return_var] = local_context.get_variable(return_var)
                    break

                # Advance program counter
                frame.index += 1

            return outputs
        finally:
            # Always pop the frame to keep stack consistent
            self._dbg.pop_frame()

    def _execute_flow_control_instruction(
        self, instruction: lc.FlowControlInstruction, context: ExecutionContext
    ) -> Any:
        """
        Wrap the base instruction execution to track recursion depth for stepping.
        """
        self._dbg.recursion_depth += 1
        try:
            return super()._execute_flow_control_instruction(instruction, context)
        finally:
            self._dbg.recursion_depth -= 1


