"""Load VTK's dynamically registered Python array/model overrides when frozen."""


def register_overrides():
    # VTK 9.7 loads these via strings in a module map and silently skips missing
    # modules. Explicit imports support both normal Python and PyInstaller's
    # importer and retain the buffer/NumPy API of implicit structured-point arrays.
    import vtkmodules.vtkCommonCore
    import vtkmodules.numpy_interface.array_overrides
    import vtkmodules.util.data_array_selection
    import vtkmodules.vtkCommonDataModel
    import vtkmodules.util.data_model
    import vtkmodules.vtkCommonExecutionModel
    import vtkmodules.util.execution_model
