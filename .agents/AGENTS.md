

- **Creating New Operations**: When asked to add a new primitive or operation to the compute graph, you MUST read `.agents/AddingNewOps.md` to understand the 5-step pipeline for frontend building, dimension collapsing, and CPU/Metal multidimensional dispatching.
- **Testing Standard**: All new mathematical ops must be verified against an exact Python equivalent using `mlx.core` (saved as a temporary `test_mlx_{op}.py`), and tested inside the `computational_graphV2` function in `MatrixH.mm`. You MUST delete the python test script after verification to avoid polluting the workspace.
