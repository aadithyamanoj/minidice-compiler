`ifndef DICE_CONFIG_VH
`define DICE_CONFIG_VH

// =========================================================
// Global architectural configuration constants
// =========================================================

`define DICE_MAX_KERNEL_SIZE         65536
// Available registers
`define DICE_GPR_NUM                 8 // General Purpose Registers
`define DICE_PR_NUM                  2  // Predicate Registers
`define DICE_CR_NUM                  8  // Constant Registers

// Available CGRA memory ports
`define DICE_CGRA_MEM_PORTS          4
`define DICE_BITSTREAM_SIZE          1074 // serial CGRA config bits

// Architectural configurations
`define DICE_ADDR_WIDTH              16
`define DICE_MAX_KERNEL_ID           1
`define DICE_MAX_GRID_SIZE           65536
`define DICE_NUM_CGRA_CLUSTERS        1
`define DICE_NUM_CGRA_CORES           1
`define DICE_NUM_MAX_THREADS_PER_CORE 16
`define DICE_NUM_MAX_CTA_PER_CORE     1
`define DICE_NUM_RETIRE_TABLE_ENTRIES 1
`define DICE_SMEM_SIZE_PER_CORE     16384  // in Bytes
`define DICE_L1_LINE_SIZE            4   // in Bytes
`define DICE_L2_LINE_SIZE            4   // in Bytes
`define DICE_L3_LINE_SIZE            4   // in Bytes

// P-graph configuration
`define DICE_MAX_PGRAPHS              32   // Maximum p-graphs per kernel
`define DICE_METADATA_WIDTH          256   // Must match line size — one metadata per cache line

// SIMT Stack configuration
`define DICE_SIMT_STACK_DEPTH        8    // Maximum SIMT stack depth per CTA

`endif // DICE_CONFIG_VH
