// PCIe IDs for the AFI manifest + shell ID registers. The HDK manifest
// generator parses these LITERAL defines (device_vendor packing, matching
// the silicon-validated FORGE AFIs: device 0xF010, vendor 0x1D0F).
`ifndef CL_ID_DEFINES
`define CL_ID_DEFINES
`define CL_SH_ID0 32'hF010_1D0F
`define CL_SH_ID1 32'h1D81_1D0F
`endif
