/*
    uses TUSER[0] to determine when a frame is ready to grab
    counts HEIGHT number of lines, determined by TLAST
	see https://docs.amd.com/r/en-US/ug934_axi_videoIP/AXI4-Stream-Signaling-Interface

	downstream modules deasserts write_enable to prevent BRAM overwrites during decoding
		- so this module completely does not care 
	WARNING: potentially long combinational path for ready, avoid somehow?
*/
`timescale 1ns / 1ps
`default_nettype none // prevents system from inferring an undeclared logic (good practice)
module frame_capture #
	(
		parameter integer C_S00_AXIS_TDATA_WIDTH	= 32,
		parameter integer C_M00_AXIS_TDATA_WIDTH	= 32,
        parameter integer HEIGHT                    = 2
	)
	(

		// custom ports
        input wire  s00_start_frame,
		output logic [C_S00_AXIS_TDATA_WIDTH-1 : 0] m00_bram_address, //oversized but whatever
		output logic m00_bram_data,
		input wire s00_start_capture,	// single cycle high that must be triggered by downstream modules to escape the IDLE state
		output logic m00_bram_wea, // need to stop module from overwriting the top left corner of the bram when waiting

		// Ports of Axi Slave Bus Interface S00_AXIS
		input wire  s00_axis_aclk, s00_axis_aresetn,
		input wire  s00_axis_tlast, s00_axis_tvalid,
		input wire [C_S00_AXIS_TDATA_WIDTH-1 : 0] s00_axis_tdata,
		input wire [(C_S00_AXIS_TDATA_WIDTH/8)-1: 0] s00_axis_tstrb
		// output logic  s00_axis_tready, commented out since module will always be ready

 
		// Ports of Axi Master Bus Interface M00_AXIS
		// we arent interfacing with axis module downstream
		// input wire  m00_axis_aclk, m00_axis_aresetn,
		// input wire  m00_axis_tready,
		// output logic  m00_axis_tvalid, m00_axis_tlast,
		// output logic [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
		// output logic [(C_M00_AXIS_TDATA_WIDTH/8)-1: 0] m00_axis_tstrb
	);
  

	enum {IDLE, FRAME_WAIT, FRAME_WRITE} state;
	logic [C_M00_AXIS_TDATA_WIDTH -1: 0] num_lines; // oversized but whatever

	always_ff @(posedge s00_axis_aclk)begin
		if (~s00_axis_aresetn)begin
			state <= FRAME_WAIT; // to ensure we arent eternally stalling, resets will guarentee a frame is written into BRAM by entering FRAME_WAIT stage
			m00_bram_address <= 0;
			m00_bram_data <= 0;
			num_lines <= 0;
			m00_bram_wea <= 0;
		end else begin
			case (state)

				IDLE: begin
					// waiting for decoding to finish to bump into FRAME_WAIT stage
					state <= s00_start_capture ? FRAME_WAIT : IDLE;
					m00_bram_address <= 0;
					m00_bram_data <= 0;
					num_lines <= 0;
					m00_bram_wea <= 0;
				end

				FRAME_WAIT: begin
					// wait for new frame to start
					if (s00_start_frame && s00_axis_tvalid) begin // theres no way start frame is high and valid isnt
						// cant miss immediate new pixel
						state <= FRAME_WRITE;
						m00_bram_wea <= s00_axis_tvalid;
						m00_bram_data <= s00_axis_tdata[0];// since its binarized we can just grab last bit
					end
				end

				FRAME_WRITE: begin
					m00_bram_wea <= s00_axis_tvalid;
					if (s00_axis_tvalid) begin
						// write data to brams, counting exactly how many lines were written
						m00_bram_data <= s00_axis_tdata[0];// since its binarized we can just grab last bit
						m00_bram_address <= m00_bram_address + 1;
						if (s00_axis_tlast) begin
							if (num_lines == HEIGHT - 1)begin
								state <= IDLE;
								// we didnt reset the write enable here so that the final pixel was logged
							end
							else begin
								num_lines <= num_lines + 1;
							end
						end
					end
				end

			endcase
		end
	end
endmodule
`default_nettype wire // prevents system from inferring an undeclared logic (good practice)