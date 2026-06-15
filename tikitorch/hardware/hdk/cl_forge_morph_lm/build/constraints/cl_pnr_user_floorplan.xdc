# =============================================================================
# N=4 replication floorplan (SILICON-VALIDATED, WNS -0.173 ns). N=4 is the
# SLR-local maximum: per-kernel ~122 URAM vs CL pblock URAM/SLR = 176/192/~320,
# so 1 (SLR0) + 1 (SLR1) + 2 (SLR2) = 4 (N=6 with 2/SLR overflows SLR0/SLR1).
# Each kernel is pinned SLR-local; its RS_DMA/RS_OCL register slices FLOAT to the
# SLR boundary (reg-to-reg Laguna crossings to the shared crossbar / OCL slice).
# Reset CLR is a false path. Convergence: -2.2 -> -1.75 -> -1.147 (RS_DMA) ->
# -0.576 (RS_OCL) -> -0.173 (reset false_path).
# SLR0=K0, SLR1=K1, SLR2=K2(lower)+K3(upper). rows: SLR0=Y0-3,SLR1=Y4-7,SLR2=Y8-11.
# =============================================================================

create_pblock pb_k0
add_cells_to_pblock pb_k0 [get_cells -hier -regexp {.*KINST\[0\]\..*}]
remove_cells_from_pblock pb_k0 [get_cells -hier -regexp {.*KINST\[0\]\.RS_(DMA|OCL).*}]
resize_pblock pb_k0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3 CLOCKREGION_X4Y0:CLOCKREGION_X7Y0}
set_property SNAPPING_MODE ON [get_pblocks pb_k0]

create_pblock pb_k1
add_cells_to_pblock pb_k1 [get_cells -hier -regexp {.*KINST\[1\]\..*}]
remove_cells_from_pblock pb_k1 [get_cells -hier -regexp {.*KINST\[1\]\.RS_(DMA|OCL).*}]
resize_pblock pb_k1 -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7 CLOCKREGION_X5Y4:CLOCKREGION_X5Y7}
set_property SNAPPING_MODE ON [get_pblocks pb_k1]

create_pblock pb_k2
add_cells_to_pblock pb_k2 [get_cells -hier -regexp {.*KINST\[2\]\..*}]
remove_cells_from_pblock pb_k2 [get_cells -hier -regexp {.*KINST\[2\]\.RS_(DMA|OCL).*}]
resize_pblock pb_k2 -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y9}
set_property SNAPPING_MODE ON [get_pblocks pb_k2]

create_pblock pb_k3
add_cells_to_pblock pb_k3 [get_cells -hier -regexp {.*KINST\[3\]\..*}]
remove_cells_from_pblock pb_k3 [get_cells -hier -regexp {.*KINST\[3\]\.RS_(DMA|OCL).*}]
resize_pblock pb_k3 -add {CLOCKREGION_X0Y10:CLOCKREGION_X7Y11}
set_property SNAPPING_MODE ON [get_pblocks pb_k3]

# Async-reset synchronizer CLR pins: recovery/removal is a false path.
set_false_path -to [get_pins -hier -filter {NAME =~ *KINST*rst_pipe_reg*/CLR}]
