# urdf2any

A pure-Python tool that converts URDF to USD / MJCF, independent of Isaac Sim and Omniverse.

## Documentation layout (follow this)

- `docs/notes.md` — long-lived knowledge: verified facts, semantics mapping tables, decision log. **When you learn something during implementation, append it to the decision log here** (date + decision + reason, a few lines).
- `docs/implementation-plan.md` — disposable TODO. Collapse completed phases to one line. Do not accumulate learnings here (move them to notes.md).
- Do not create new documentation files. Keep everything in the two files above plus the README.
- As implementation progresses, replace notes.md descriptions that are already expressed in code/tests with "see code" references, keeping it lean.

## Rules

- The IR uses SI units and radians. Unit conversion is the backend's responsibility (USD uses degrees).
- Clone destinations of the reference implementations are in notes.md §11.
