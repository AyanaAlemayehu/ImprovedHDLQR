import cocotb
import os
import random
import sys
import logging
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, ReadOnly,with_timeout, First, Join 

#cocotb bus
from cocotb_bus.bus import Bus
from cocotb_bus.drivers import BusDriver
from cocotb_bus.monitors import Monitor
from cocotb_bus.monitors import BusMonitor
from cocotb_bus.scoreboard import Scoreboard

#misc
import numpy as np
import struct
# import matplotlib.pyplot as plt

from cocotb.handle import SimHandleBase

TRUE = ['true', 'True', 't', '1', True]
FALSE = ['false', 'False', 'f', '0', False]

class CustomScoreboard(Scoreboard):

    def __init__(self, dut = None, fail_immediately=False, reorder_depth=0, verbose = False):
        super().__init__(dut=dut, fail_immediately=fail_immediately, reorder_depth=reorder_depth)
        if (verbose):
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

    def compare(self, got, exp, log, strict_type=True):
        # check if more keys need to be added to custom 
        custom = ["data", "name"]
        for k, v in exp.items():
            if k not in custom and got[k] != exp[k]:
                self.errors += 1
                log.error(f"Received k, v pair: ({k}, {got[k]}) differed from expected k, v pair ({k}, {exp[k]})")
                if (self._imm):
                    assert False, f"Received k, v pair: ({k}, {got[k]}) differed from expected k, v pair ({k}, {exp[k]})"
            if k == 'data':
                # implement to check received data correctness, otherwise pass if not necessary
                assert got[k].integer == int(exp[k]), f"Received {got[k].integer} expected {int(exp[k])}"
class Tester:
    """
    Default Tester for one slave input and one master output
    """
    def __init__(self, dut_entity: SimHandleBase, debug=False, verbose=False, history=False):
        self.dut = dut_entity
        self.verbose = verbose
        self.log = logging.getLogger(f"cocotb.Tester.{dut_entity.name}")
        if (self.verbose):
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)
        self.input_mon = AXISMonitor(self.dut,'s00',self.dut.s00_axis_aclk, callback=self.model, verbose=verbose)
        self.output_mon = AXISMonitor(self.dut,'m00',self.dut.s00_axis_aclk, verbose=verbose, history=history)
        self.input_driver = AXISDriver(self.dut,'s00',self.dut.s00_axis_aclk)
        self._checker = None
        # Create a scoreboard on the stream_out bus
        self.expected_output = [] #contains list of expected outputs (Growing)
        self.scoreboard = CustomScoreboard(dut=self.dut, fail_immediately=False, verbose=verbose)
        self.scoreboard.add_interface(self.output_mon, self.expected_output)
 
    def start(self) -> None:
        self._checker = True
        self.input_mon.start()
        self.output_mon.start()
        self.input_driver.start()

    def stop(self) -> None:
        """Stops everything"""
        if self._checker is None:
            raise RuntimeError("Monitor never started")
        self.input_mon.stop()
        self.output_mon.stop()
        self.input_driver.stop()

    def model(self, transaction):
        #define a model here, otherwise self.expected_output.append(transaction) passthrough if not necessary
        tran_copy = transaction.copy()
        inp = str(self.input_mon.bus.axis_tdata.value)
        # indexing is flipped
        r = int(inp[2:12], 2)
        b = int(inp[12:22], 2)
        g = int(inp[22:32], 2)
        y = .2126*r + .7156*g + .0722*b
        # print("y and thresh", y, self.dut.thresh_in.value.integer)
        if (y < self.dut.thresh_in.value.integer):
            tran_copy['data'] = 0
        else:
            tran_copy['data'] = 4294967295

        # print(, "the input value")
        #   tran_copy['data'] = 
        self.expected_output.append(tran_copy)

    # potentially useful utility functions (havent checked them)
    def twosC(self, v):
        """ assumes a binary string v without the 0b beginning that is in 16bit twos complement form"""
        val = int("0b" + v[1:], 2) - (int(v[0]) << 15)
        return val
 
    # def plot_result(self,length, base=[]):
    #     input_vals = self.input_mon.values #array I built up over time (could use for comparing)
    #     output_vals = np.array([int(v['data']) if 'x' not in v['data'] else int(0) for v in self.output_mon.values])
    #     top = ((output_vals>>16)&0xFFFF).astype(np.int16)
    #     bott = (output_vals&0xFFFF).astype(np.int16)
    #     # print(top) #for sanity checking
    #     # print(bott) #for sanity checking
    #     if (len(base) > 0):
    #         plt.plot(length, base)
    #     plt.plot(length, top)
    #     plt.plot(length, bott)
    #     plt.show()

class AXISMonitor(BusMonitor):
    """
    monitors axi streaming bus
    """
    transactions = 0
    def __init__(self, dut, name, clk, callback=None, verbose=False, history=False):
        self._checker = None
        self._signals = ['axis_tvalid','axis_tready','axis_tlast','axis_tdata','axis_tstrb']
        BusMonitor.__init__(self, dut, name, clk, callback=callback)
        self.clock = clk
        self.transactions = 0
        self.verbose = verbose
        self.history = history
        self.values = []
        self.log = logging.getLogger(f"cocotb.AXISMonitor.{dut.name}")
        if (self.verbose):
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

    async def _monitor_recv(self):
        """
        Monitor receiver
        """
        rising_edge = RisingEdge(self.clock) # make these coroutines once and reuse
        falling_edge = FallingEdge(self.clock)
        read_only = ReadOnly() #This is
        while True:
            await rising_edge
            await read_only  #readonly (the postline)
            if (self._checker):
                valid = self.bus.axis_tvalid.value
                ready = self.bus.axis_tready.value
                last = self.bus.axis_tlast.value
                data = self.bus.axis_tdata.value
                if valid and ready:
                    self.transactions += 1
                    thing = dict(data=data,last=last,name=self.name,count=self.transactions)
                    self._recv(thing)
                    if (self.history):
                        self.values.append(thing)
                    if (self.verbose):
                        self.log.debug(thing)
        
    def start(self) -> None:
        self._checker = True

    def stop(self) -> None:
        """Stops everything"""
        self._checker = False


class AXISDriver(BusDriver):
  def __init__(self, dut, name, clk, verbose=False):
    self._signals = ['axis_tvalid', 'axis_tready', 'axis_tlast', 'axis_tdata','axis_tstrb']
    BusDriver.__init__(self, dut, name, clk)
    self.clock = clk
    self.bus.axis_tdata.value = 0
    self.bus.axis_tstrb.value = 0
    self.bus.axis_tlast.value = 0
    self.bus.axis_tvalid.value = 0
    self._checker = None
    self.verbose = verbose
    self.log = logging.getLogger(f"cocotb.AXISDriver.{dut.name}")
    if (self.verbose):
        self.log.setLevel(logging.DEBUG)
    else:
        self.log.setLevel(logging.WARNING)

  def start(self) -> None:
    self._checker = True

  def stop(self) -> None:
    """Stops everything"""
    self._checker = False

  async def _driver_send(self, value, sync=True):
    if (not self._checker):
        return
    
    rising_edge = RisingEdge(self.clock) # make these coroutines once and reuse
    falling_edge = FallingEdge(self.clock)
    read_only = ReadOnly() #This is    
    if "type" not in value:
        raise Exception("type missing in value")
    if value["type"] == "single":
        await falling_edge
        if (self.bus.axis_tready.value != 1):
            await RisingEdge(self.bus.axis_tready)
        await falling_edge
        self.bus.axis_tstrb.value = value["contents"]["strb"]
        self.bus.axis_tdata.value = value["contents"]["data"]
        self.bus.axis_tlast.value = value["contents"]["last"]
        self.bus.axis_tvalid.value = 1
        await falling_edge
        self.bus.axis_tvalid.value = 0


    elif value["type"] == "burst":
        temp = np.copy(value["contents"]["data"])
        ind = 0
        #await falling_edge
        while ind < len(temp):

            if (self.bus.axis_tready.value != 1):
                await RisingEdge(self.bus.axis_tready)
            await falling_edge
            self.bus.axis_tdata.value = temp[ind].item()
            self.bus.axis_tstrb.value = cocotb.binary.BinaryValue(value = 15, n_bits = self.bus.axis_tstrb.value.n_bits)
            if (ind == len(temp) - 1):
                self.bus.axis_tlast.value = 1
            self.bus.axis_tvalid.value = 1 
            if (ind == len(temp) - 1):
                await falling_edge
                self.bus.axis_tvalid.value = 0
                self.bus.axis_tlast.value = 0
            ind += 1

    else:
        raise Exception("invalid value type")

async def set_ready(clk_wire, dut, val):
    await FallingEdge(clk_wire)
    dut.m00_axis_tready.value = val

async def reset(clk_wire, rst_wire, cycles, value):
    rst_wire.value = value
    await ClockCycles(clk_wire, cycles + 1)
    rst_wire.value = value ^ 1

@cocotb.test()
async def test_a(dut, *args):
    """default cocotb test"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)

    # configure the module
    dut.thresh_in.value = 1

    # feed the driver
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 100761696}}

    #send the data
    tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 100) #figure out how many cycles to wait for a result

    # change configuration
    dut.thresh_in.value = 50

    # feed the driver
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 100761696}}

    #send the data
    tester.input_driver.append(data)

    # wait and apply backpressure ensuring pipeline freezes when ready is deasserted
    await ClockCycles(dut.s00_axis_aclk, 2)
    await set_ready(dut.s00_axis_aclk, dut, 0)
    await ClockCycles(dut.s00_axis_aclk, 23) #figure out how many cycles to wait for a result

    await set_ready(dut.s00_axis_aclk, dut, 1)
    await ClockCycles(dut.s00_axis_aclk, 25) #figure out how many cycles to wait for a result

    assert tester.input_mon.transactions==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result


"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
 
def runner():
    """Simulate the counter using the Python runner."""
    run_test_args = sys.argv[1:]
    print(f"Running with these arguments: {run_test_args}")
    for arg in run_test_args:
        os.environ[arg] = 'True'
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "stream_monocolor.sv"] #grow/modify this as needed.
    # i believe you can add another source like below?
    sources.append(proj_path / "hdl" / "rgb_to_ycrcb.sv")
    build_test_args = ["-Wall"]#,"COCOTB_RESOLVE_X=ZEROS"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="monocolor",
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True
    )
    runner.test(
        hdl_toplevel="monocolor",
        test_module="test_stream_monocolor",
        test_args=run_test_args,
        waves=True
    )
 
if __name__ == "__main__":
    runner()
