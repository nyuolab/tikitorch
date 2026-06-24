# cl_timing_user.xdc — INTENTIONALLY EMPTY (this file is NOT read by the F2 build).
#
# The AWS F2 HDK aws_build_dcp_from_cl flow reads ONLY ${SHELL_MODE}_cl_pnr_user.xdc
# from the CL constraints dir (= small_shell_cl_pnr_user.xdc on the small shell;
# CONFIRMED in aws_build.full.log line 7265-7267). It does NOT read cl_timing_user.xdc
# or cl_synth_user.xdc at all. An earlier set_clock_groups -asynchronous lived here and
# was a SILENT NO-OP for builds v9/v10 (which is why their worst paths were byte-identical:
# the constraint never reached Vivado, leaving WARNING [Timing 38-164] unaddressed).
#
# ALL CL user constraints (the async clock groups AND the floorplan pblock) now live in
# small_shell_cl_pnr_user.xdc — the only file this flow actually parses.
