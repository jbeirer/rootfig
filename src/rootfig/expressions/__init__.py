"""Expression strings over branches.

rootfig expressions use ordinary Python syntax and are evaluated with NumPy
and Awkward Array semantics, so ``Muon_pt > 20`` produces a boolean array
with the same (possibly jagged) structure as ``Muon_pt``.

Supported syntax
----------------
* Names refer to branches (``Muon_pt``). Branch names that are not valid
  Python identifiers are written in backticks: ```jet1_b-tag` > 0.5``.
* Arithmetic ``+ - * / // % **`` and unary ``-``.
* Comparisons ``== != < <= > >=`` including chained comparisons
  (``20 < Muon_pt < 100``), which are rewritten element-wise.
* Boolean combinations with ``&``, ``|``, ``~`` **or** with ``and``, ``or``,
  ``not``; the keyword forms are rewritten to the element-wise operators.
* Indexing and slicing (``Muon_pt[:, 0]``).
* A fixed set of functions (see :data:`FUNCTIONS`): element-wise NumPy
  functions such as ``abs``, ``sqrt``, ``log``, ``where``, and per-event
  reductions over jagged branches: ``count``, ``sum``, ``min``, ``max``,
  ``mean``, ``any``, ``all``, ``first``.
* Constants ``pi``, ``e``, ``inf``, ``nan``, ``True``, ``False``.

Everything else (attribute access, lambdas, comprehensions, string literals,
calls to unknown functions) is rejected at parse time with
:class:`~rootfig.errors.ExpressionError`.

The public entry points are :func:`parse`, :class:`Expression`, and
:func:`evaluate`.
"""

from rootfig.expressions.functions import CONSTANTS, FUNCTIONS
from rootfig.expressions.parser import Expression, ExpressionLike, evaluate, parse

__all__ = ["CONSTANTS", "FUNCTIONS", "Expression", "ExpressionLike", "evaluate", "parse"]
