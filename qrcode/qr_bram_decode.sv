`timescale 1ns / 1ps
`default_nettype none

/*
  List of known optimizations:
  1. horizontal and vertical pattern finder can be run in parallel (speed up by almost 2x)
  2. somewhere I believe I am waiting two clock cycles for every pixel lookup, when i could actually just make my logic work two cycles behind


  Notes:
  1. module resetting needs to be done, maybe I can use a start_decode signal to reset the entire ip then start
*/

typedef enum logic [3:0] {HORIZ_PATTERNS, VERT_PATTERNS, CLEAN, BOUNDS, CROSS, FIND_MOD, DOWNSAMPLE_0, DOWNSAMPLE_1, DOWNSAMPLE_2, UNMASK, FINISHED, IDLE} fsm_state;

module qr_decoder(
  // general ports
  input wire axis_clk,
  input wire start_decode,
  input wire resetn,// active low
  output logic finished_decode,

  // bram ports
  input wire bram_data,
  output logic [19:0] bram_address,

  // qr decoded values
  output logic [159:0] qr_out,

  // sanity/debugging ports
  output fsm_state outstate,
  output logic err_ce1,
  output logic err_ce2

  );


  // dumbness

  /*
   PARAMETER INITIALIZATION
   ---------------------------------------------------------------------
  */
    localparam STORED_WIDTH = 640;
    localparam STORED_HEIGHT = 480;
    localparam QR_SIZE = 21;


  logic [19:0] baddress;
  logic bdata;
  logic sys_rst; // active high
  logic err_found;
  logic ce1, ce2;
  assign sys_rst = (~resetn) || start_decode;
  assign err_ce1 = ce1;
  assign err_ce2 = ce2;
  assign outstate = state;
  assign bram_address = baddress;
  assign bdata = bram_data;

  initial begin
    sys_rst = 0;
    err_found = 0;
  end

  // error tracking
  always_ff @(posedge axis_clk) begin
    if (sys_rst) begin
      err_found <= 0;
    end
    else begin
      if (ce1 || ce2)begin
        err_found <= 1;
      end
    end
  end
  /*
    Top Level State Maching
  */
    fsm_state state = HORIZ_PATTERNS; // check here for errors

    always_ff @(posedge axis_clk) begin

      if (~resetn || start_decode) begin
          // either whole thing is reset or (more likely) we've been triggered to decode
          state <= HORIZ_PATTERNS;
          finished_decode <= 0;
      end else if (err_found) begin
          // quit out
          finished_decode <= 1;
          state <= IDLE;
      end else begin
          sys_rst <= 0;
          case (state)
          HORIZ_PATTERNS: begin
                // detecting the horizontal patterns
                state <= BRAM_one_horizontal_data_valid == 1'b1 ? VERT_PATTERNS : HORIZ_PATTERNS;
          end
          VERT_PATTERNS: begin
                // detecting the vertical patterns
                state <= BRAM_one_vertical_data_valid == 1'b1 ? CLEAN : VERT_PATTERNS;
          end

          CLEAN: begin
                state <= (clean_horz_valid_saved && clean_vert_valid_saved)? BOUNDS: CLEAN;
          end

          BOUNDS: begin
               state <= (valid_bound) ? CROSS : BOUNDS;
          end

          CROSS: begin
                state <= (cross_valid)? FIND_MOD: CROSS;
          end

          FIND_MOD: begin
                state <= (mod_size_valid)? DOWNSAMPLE_0: FIND_MOD;
          end

          DOWNSAMPLE_0: begin
                state <= (valid_qr_0)? DOWNSAMPLE_1: DOWNSAMPLE_0;
          end

          DOWNSAMPLE_1: begin
                state <= (valid_qr_1)? DOWNSAMPLE_2: DOWNSAMPLE_1;
          end

          DOWNSAMPLE_2: begin
                state <= (valid_qr_2)? UNMASK: DOWNSAMPLE_2;
          end

          UNMASK: begin
                state <= (unmask_ready)? FINISHED: UNMASK;
          end 

          FINISHED: begin
                state <= IDLE;
                finished_decode <= 1;
          end

          IDLE: begin
              finished_decode <= 0;
          end

          endcase
      end
    end 

  // pattern ratio finders
  logic BRAM_one_horizontal_pixel_data;
  logic [19:0] BRAM_one_horizontal_pixel_address;
  logic [479:0] BRAM_one_horizontal_finder_encodings;
  logic BRAM_one_horizontal_data_valid;

  horizontal_pattern_ratio_finder #(.WIDTH(STORED_WIDTH), .HEIGHT(STORED_HEIGHT)) 
    horizontal
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .pixel_data(BRAM_one_horizontal_pixel_data),// MAKE NEW VARIABLE
        .start_finder((state == HORIZ_PATTERNS)),
        .pixel_address(BRAM_one_horizontal_pixel_address),
        .finder_encodings(BRAM_one_horizontal_finder_encodings),
        .data_valid(BRAM_one_horizontal_data_valid)
    );

  logic BRAM_one_vertical_pixel_data;
  logic [19:0] BRAM_one_vertical_pixel_address;
  logic [479:0] BRAM_one_vertical_finder_encodings;
  logic BRAM_one_vertical_data_valid;

  vertical_pattern_ratio_finder vertical
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .pixel_data(BRAM_one_vertical_pixel_data),// MAKE NEW VARIABLE
        .start_finder(BRAM_one_horizontal_data_valid),
        .pixel_address(BRAM_one_vertical_pixel_address),
        .finder_encodings(BRAM_one_vertical_finder_encodings),
        .data_valid(BRAM_one_vertical_data_valid)
    );


  // cleaning found patterns
    logic clean_horz_valid;
    logic clean_vert_valid;
    logic clean_horz_valid_saved;
    logic clean_vert_valid_saved;

    always_ff @(posedge axis_clk) begin
      if (sys_rst) begin
          clean_horz_valid_saved <= 1'b0;
          clean_vert_valid_saved <= 1'b0;
      end
      else begin
        if (clean_horz_valid)
          clean_horz_valid_saved <=1'b1;

        if (clean_vert_valid)
          clean_vert_valid_saved <=1'b1;
      end
    end

    logic [479:0] BRAM_one_vertical_finder_encodings_clean;
    logic [479:0] BRAM_one_horizontal_finder_encodings_clean;


    clean_patterns #(.WIDTH(STORED_WIDTH))
    clean_horz
    (   
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .pattern(BRAM_one_horizontal_finder_encodings),
        .start_cleaning(BRAM_one_horizontal_data_valid),
        .data_valid(clean_horz_valid),
        .clean_pattern(BRAM_one_horizontal_finder_encodings_clean)
    );

    clean_patterns #(.WIDTH(STORED_WIDTH))
    clean_vert
    (   
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .pattern(BRAM_one_vertical_finder_encodings),
        .start_cleaning(BRAM_one_vertical_data_valid),
        .data_valid(clean_vert_valid),
        .clean_pattern(BRAM_one_vertical_finder_encodings_clean)
    );

  // bounds determination
    logic [8:0] bounds_x [1:0];
    logic [8:0] bounds_y [1:0];
    logic valid_bound;

    bounds #(.WIDTH(STORED_WIDTH), .HEIGHT(STORED_HEIGHT),
             .OFFSET(10))
    bounds_mod
    (
      .clk_in(axis_clk),
      .rst_in(sys_rst),
      .horz_patterns(BRAM_one_horizontal_finder_encodings_clean),
      .vert_patterns(BRAM_one_vertical_finder_encodings_clean),
      .start_bound(clean_horz_valid_saved && clean_vert_valid_saved),

      .bound_x(bounds_y), // ERROR: SWITCHED THEM HERE (cause they were actually inverted) (can also just switch the finder encodings)
      .bound_y(bounds_x),
      .valid_bound(valid_bound)
    );

  // cross referencing finder pattern encodings with bram data
  // only current place that can err out
  logic BRAM_one_cross_reading_pixel;
  logic [19:0] BRAM_one_cross_reading_address;
  logic [8:0] centers_x_cross [2:0];
  logic [8:0] centers_y_cross [2:0];
  logic cross_valid;

  cross_patterns cross_mod
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .horz_patterns(BRAM_one_vertical_finder_encodings_clean), // ERROR: SWITCHED AGAIN HERE 
        .vert_patterns(BRAM_one_horizontal_finder_encodings_clean),
        .start_cross(valid_bound),
        .pixel_reading(BRAM_one_cross_reading_pixel),
        .bound_x(bounds_x),
        .bound_y(bounds_y),

        .address_reading(BRAM_one_cross_reading_address),
        .centers_x(centers_x_cross),
        .centers_y(centers_y_cross),
        .centers_valid(cross_valid),
        .centers_not_found_error(ce1),// using rgb to err out
        .centers_not_found_error2(ce2)
    );

  logic [8:0] module_size;
  logic mod_size_valid;
  find_mod_size #(.MODULES(14))// 15 modules between version 1 qr codes
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .centers_x(centers_x_cross),
        .centers_y(centers_y_cross),
        .start_downsample(cross_valid),

        .mod_size(module_size), // oversized by a lot lol
        .mod_size_valid(mod_size_valid)
    );

    logic BRAM_one_downsample_reading_pixel_0;
    logic [19:0] BRAM_one_downsample_address_0;
    logic [440:0] qr_code_0;
    logic valid_qr_0;

    downsample_0 #(.WIDTH(STORED_WIDTH))
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .start_downsample(mod_size_valid),
        .reading_pixel(BRAM_one_downsample_reading_pixel_0),
        .module_size(module_size),
        .centers_x(centers_x_cross),
        .centers_y(centers_y_cross),

        .reading_address(BRAM_one_downsample_address_0),
        .qr_code(qr_code_0),
        .valid_qr(valid_qr_0)
    );

    logic BRAM_one_downsample_reading_pixel_1;
    logic [19:0] BRAM_one_downsample_address_1;
    logic [440:0] qr_code_1;
    logic valid_qr_1;

    downsample_1 #(.WIDTH(STORED_WIDTH))
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .start_downsample(valid_qr_0),
        .reading_pixel(BRAM_one_downsample_reading_pixel_1),
        .module_size(module_size),
        .centers_x(centers_x_cross),
        .centers_y(centers_y_cross),

        .reading_address(BRAM_one_downsample_address_1),
        .qr_code(qr_code_1),
        .valid_qr(valid_qr_1)
    );

    logic BRAM_one_downsample_reading_pixel_2;
    logic [19:0] BRAM_one_downsample_address_2;
    logic [440:0] qr_code_2;
    logic valid_qr_2;

    downsample_2 #(.WIDTH(STORED_WIDTH))
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .start_downsample(valid_qr_1),
        .reading_pixel(BRAM_one_downsample_reading_pixel_2),
        .module_size(module_size),
        .centers_x(centers_x_cross),
        .centers_y(centers_y_cross),

        .reading_address(BRAM_one_downsample_address_2),
        .qr_code(qr_code_2),
        .valid_qr(valid_qr_2)
    );

  logic [440:0] qr_code;


  downsample_combine #(.CODE_SIZE(QR_SIZE))
      (
          .qr_0(qr_code_0),
          .qr_1(qr_code_1),
          .qr_2(qr_code_2),
          .qr_code(qr_code)
      );


  logic [440:0] qr_code_unmask;
  logic unmask_ready;


  unmask #(.MOD_SIZE(QR_SIZE))
    (
        .clk_in(axis_clk),
        .rst_in(sys_rst),
        .start_unmask(valid_qr_2),
        .downsampled_qr(qr_code),
        .qr_unmasked(qr_code_unmask),
        .unmask_ready(unmask_ready)
    );

  logic [3:0] data_type;
  logic [7:0] data_length;
  logic [7:0] bytes [18:0];

  assign qr_out = {bytes[0], bytes[1], bytes[2], bytes[3], 
                   bytes[4], bytes[5], bytes[6], bytes[7],
                   bytes[8], bytes[9], bytes[10], bytes[11],
                   bytes[12], bytes[13], bytes[14], bytes[15],
                   bytes[16], bytes[17], bytes[18], 8'b0};
  decode #(.MOD_SIZE(QR_SIZE))
    (
        .qr_unmasked(qr_code_unmask),
        .data_type(data_type),
        .data_length(data_length),
        .bytes(bytes)
    );

  /*
    Controling Memory Ports
  */
    always_comb begin
      if (state == HORIZ_PATTERNS)begin
        // reading frame buffer goes to horizontal pattern finder if state is HORIZ_PATTERNS
        baddress = BRAM_one_horizontal_pixel_address;
        BRAM_one_horizontal_pixel_data = bdata;
      end
      else if (state == VERT_PATTERNS) begin
        baddress = BRAM_one_vertical_pixel_address;
        BRAM_one_vertical_pixel_data = bdata;
      end
      else if (state == CROSS) begin
        baddress = BRAM_one_cross_reading_address;
        BRAM_one_cross_reading_pixel = bdata;
      end
      else if (state == DOWNSAMPLE_0) begin
        baddress = BRAM_one_downsample_address_0;
        BRAM_one_downsample_reading_pixel_0 = bdata;
      end
      else if (state == DOWNSAMPLE_1) begin
        baddress = BRAM_one_downsample_address_1;
        BRAM_one_downsample_reading_pixel_1 = bdata;
      end
      else if (state == DOWNSAMPLE_2) begin
        baddress = BRAM_one_downsample_address_2;
        BRAM_one_downsample_reading_pixel_2 = bdata;
      end
      else begin
        baddress = 0;
      end
      // else if (state == FINISHED) begin
          // nothing to do
      // end
    end

endmodule // top_level
`default_nettype wire
