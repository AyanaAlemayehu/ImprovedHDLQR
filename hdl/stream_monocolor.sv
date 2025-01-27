/*
    converts RGB 10bit stream into monocolor using 
*/

`timescale 1ns / 1ps
`default_nettype none // prevents system from inferring an undeclared logic (good practice)
module monocolor #
	(
		parameter integer C_S00_AXIS_TDATA_WIDTH	= 32,
		parameter integer C_M00_AXIS_TDATA_WIDTH	= 32
	)
	(
        // custom ports
        input wire [9:0] thresh_in,

        // vivado extra ports
        input wire s_axis_video_TDEST,
        input wire s_axis_video_TID,
        input wire [3:0] s_axis_video_TKEEP,
        input wire s_axis_video_TUSER,

        output wire m_axis_video_TDEST,
        output wire m_axis_video_TID,
        output wire [3:0] m_axis_video_TKEEP,
        output wire m_axis_video_TUSER,

		// Ports of Axi Slave Bus Interface S00_AXIS
		input wire  s00_axis_aclk, s00_axis_aresetn,
		input wire  s00_axis_tlast, s00_axis_tvalid,
		input wire [C_S00_AXIS_TDATA_WIDTH-1 : 0] s00_axis_tdata,
		input wire [(C_S00_AXIS_TDATA_WIDTH/8)-1: 0] s00_axis_tstrb,
		output logic  s00_axis_tready,
 
		// Ports of Axi Master Bus Interface M00_AXIS
		input wire  m00_axis_aclk, m00_axis_aresetn,
		input wire  m00_axis_tready,
		output logic  m00_axis_tvalid, m00_axis_tlast,
		output logic [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
		output logic [(C_M00_AXIS_TDATA_WIDTH/8)-1: 0] m00_axis_tstrb
	);
  

    //output of rgb to ycrcb conversion
    logic [9:0] y_full; //y component of y cr cb conversion of full pixel
    
    // pipelines for latency
    logic [2:0] valid_pipe;
    logic [2:0] tlast_pipe;
    logic [2:0] [(C_S00_AXIS_TDATA_WIDTH/8)-1: 0] tstrb_pipe;
    logic [2:0] tdest_pipe;
    logic [2:0] tid_pipe;
    logic [2:0] [3:0] tkeep_pipe;
    logic [2:0] tuser_pipe;

    // error checking pipeline
    logic [2:0] [C_M00_AXIS_TDATA_WIDTH-1 : 0] tdata_pipe;

    // has 3 cycle latency
    // assuming 10bit RGB MSB aligned with 2 bits of zero padding
    rgb_to_ycrcb rgbtoycrcb_m(
    .clk_in(s00_axis_aclk),
    .r_in(s00_axis_tdata[29:20]),
    .g_in(s00_axis_tdata[9:0]),
    .b_in(s00_axis_tdata[19:10]),
    .y_out(y_full)
    );

  assign s00_axis_tready = m00_axis_tready;
  assign m00_axis_tvalid = valid_pipe[2];
  assign m00_axis_tlast = tlast_pipe[2];
  assign m00_axis_tdata = y_full < thresh_in ? 32'h00000000 : 32'hFFFFFFFF;
//   assign m00_axis_tdata = tdata_pipe[2]; // checking if pass through will work
  assign m00_axis_tstrb = tstrb_pipe[2];
  assign m_axis_video_TDEST = tdest_pipe[2];
  assign m_axis_video_TID = tid_pipe[2];
  assign m_axis_video_TKEEP = tkeep_pipe[2];
  assign m_axis_video_TUSER = tuser_pipe[2];
  always_ff @(posedge s00_axis_aclk)begin
    if (~s00_axis_aresetn) begin
        valid_pipe <= 0;
        tlast_pipe <= 0;
        tstrb_pipe[0] <= 0;
        tstrb_pipe[1] <= 0;
        tstrb_pipe[2] <= 0;
        tdest_pipe <= 0;
        tid_pipe <= 0;
        tkeep_pipe[0] <= 0;
        tkeep_pipe[1] <= 0;
        tkeep_pipe[2] <= 0;
        tuser_pipe <= 0;
        tdata_pipe[0] <= 0;
        tdata_pipe[1] <= 0;
        tdata_pipe[2] <= 0;
    end else begin
        if (s00_axis_tready) begin
            valid_pipe[0] <= s00_axis_tvalid;
            tlast_pipe[0] <= s00_axis_tlast;
            tstrb_pipe[0] <= s00_axis_tstrb;// may need to set to 4'hF
            tdest_pipe[0] <= s_axis_video_TDEST;
            tid_pipe[0] <= s_axis_video_TID;
            tkeep_pipe[0] <= s_axis_video_TKEEP;
            tuser_pipe[0] <= s_axis_video_TUSER;
            tdata_pipe[0] <= s00_axis_tdata;
            for (int i=1; i<3; i = i+1) begin
                valid_pipe[i] <= valid_pipe[i-1];
                tlast_pipe[i] <= tlast_pipe[i-1];
                tstrb_pipe[i] <= tstrb_pipe[i-1];
                tdest_pipe[i] <= tdest_pipe[i-1];
                tid_pipe[i] <= tid_pipe[i-1];
                tkeep_pipe[i] <= tkeep_pipe[i-1];
                tuser_pipe[i] <= tuser_pipe[i-1];
                tdata_pipe[i] <= tdata_pipe[i-1];
                end
            end
        end
    end
endmodule
`default_nettype wire // prevents system from inferring an undeclared logic (good practice)