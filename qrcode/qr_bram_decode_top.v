
`timescale 1 ns / 1 ps

	module qr_bram_decode_v1_0 #
	(
		// Users to add parameters here
		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface S00_AXIS

	)
	(

        // general ports
        input wire axis_clk,
        input wire start_decode,
        input wire resetn,// active low
        output wire finished_decode,

        // bram ports
        input wire bram_data,
        output wire [19:0] bram_address,

        // qr decoded values
        output wire [159:0] qr_out,

        // sanity/debugging ports
        output wire [3:0] outstate,
        output wire err_ce1,
        output wire err_ce2
	);
// Instantiation of Axi Bus Interface S00_AXIS
	qr_decoder inst (
        .axis_clk(axis_clk),
        .start_decode(start_decode),
        .resetn(resetn),
        .finished_decode(finished_decode),
        .bram_data(bram_data),
        .bram_address(bram_address),
        .qr_out(qr_out),
        .outstate(outstate),
        .err_ce1(err_ce1),
        .err_ce2(err_ce2)
	);

	// Add user logic here

	// User logic ends

	endmodule
