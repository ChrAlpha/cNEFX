# Writes build-info.json at build time: -DSOURCE, -DINPUT, -DOUTPUT, -DCONFIG.
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${SOURCE}"
  OUTPUT_VARIABLE CNEFX_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git status --porcelain WORKING_DIRECTORY "${SOURCE}"
  OUTPUT_VARIABLE cnefx_status ERROR_QUIET)
set(CNEFX_DIRTY false)
if(cnefx_status)
  set(CNEFX_DIRTY true)
endif()
set(CNEFX_CONFIG "${CONFIG}")
# configure_file rewrites the output only when its content changes.
configure_file("${INPUT}" "${OUTPUT}" @ONLY)
