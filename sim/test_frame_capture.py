"""
FOR THE SAKE OF RUNNING THIS TEST IN A REASONABLE TIME, YOU MUST SET THE HEIGHT PARAMETER TO SOMETHING SMALL LIKE
20 AND MAKE IT THE SAME IN THIS FILES PARAMETER


"""


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
# import matplotlib.pyplot as plt <--- this import doesnt work for some reason

from cocotb.handle import SimHandleBase

TRUE = ['true', 'True', 't', '1', True]
FALSE = ['false', 'False', 'f', '0', False]
LINE_LENGTH = 640 # we're doing 640xHEIGHT
HEIGHT = 2

#TRACKED SIGNALS (default to AXI Stream)
SIGNALS = ['axis_tvalid','axis_tready','axis_tlast','axis_tdata','axis_tstrb', 'start_frame', 'bram_address', 'bram_data', 'start_capture', 'bram_wea', 'axis_aresetn']

class CustomScoreboard(Scoreboard):

    def __init__(self, dut = None, fail_immediately=False, reorder_depth=0, verbose = False):
        super().__init__(dut=dut, fail_immediately=fail_immediately, reorder_depth=reorder_depth)
        if (verbose):
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

    def compare(self, got, exp, log, strict_type=True):
        # check if more keys need to be added to custom
        #print("check", got, exp) 
        custom = ["name", "reset"]
        for k, v in exp.items():
            if k not in custom and got[k] != exp[k]:
                self.errors += 1
                log.error(f"Received k, v pair: ({k}, {got[k]}) differed from expected k, v pair ({k}, {exp[k]})")
                if (self._imm):
                    assert False, f"Received k, v pair: ({k}, {got[k]}) differed from expected k, v pair ({k}, {exp[k]})"

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

        # custom state variables
        self.addr = 0
        self.data = 0
        self.num_lines = 0
        self.wea = 0
        self.state = "IDLE"
        self.decrement = 0 #for transaction counting purposes

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
        # print("state", self.state, self.wea)

        # had to put it here to catch the num_lines increment
        if (self.num_lines == HEIGHT):
            self.state = "IDLE"
            self.wea = 0
        # state variables
        if (transaction["reset"] == 0):
            self.addr = 0
            self.num_lines = 0
            self.data = 0
            self.wea = 0
            self.state = "WAIT"
        else:
            if (self.state == "IDLE"):
                self.addr = 0
                self.num_lines = 0
                self.data = 0
                self.wea = 0
                if (transaction["start_capture"] == 1):
                    self.state = "WAIT"
            elif (self.state == "WAIT"):
                if (transaction["start_frame"] == 1):
                    self.state = "WRITE"
                    self.wea = 1
                    self.data = 1 if (transaction["data"] == 4294967295) else 0
            else:
                # in write state, track 640*numlines pixels
                # note the minus one in addr at expected_output append for logics sake
                self.addr += 1
                self.data = 1 if (transaction["data"] == 4294967295) else 0
                if (transaction["last"] == 1):
                    self.num_lines += 1
                
        # transaction dump
        if ("donttrack" not in transaction):
            if (self.wea == 1): # allows us to reset internal model without adding to transaction count
                self.expected_output.append(dict(data=self.data,name="m00",count=transaction["count"] - self.decrement, addr=self.addr, wea=self.wea))
            else:
                self.decrement += 1

    # potentially useful utility functions (havent checked them)
    def twosC(self, v):
        """ assumes a binary string v without the 0b beginning that is in 16bit twos complement form"""
        val = int("0b" + v[1:], 2) - (int(v[0]) << 15)
        return val
 
    def plot_result(self,length, base=[]):
        pass
        # broken given failed matplotlib import above
        # input_vals = self.input_mon.values #array I built up over time (could use for comparing)
        # output_vals = np.array([int(v['data']) if 'x' not in v['data'] else int(0) for v in self.output_mon.values])
        # top = ((output_vals>>16)&0xFFFF).astype(np.int16)
        # bott = (output_vals&0xFFFF).astype(np.int16)
        # # print(top) #for sanity checking
        # # print(bott) #for sanity checking
        # if (len(base) > 0):
        #     plt.plot(length, base)
        # plt.plot(length, top)
        # plt.plot(length, bott)
        # plt.show()


class AXISMonitor(BusMonitor):
    """
        monitors axi streaming bus
    """
    transactions = 0
    def __init__(self, dut, name, clk, callback=None, verbose=False, history=False):
        self._checker = None
        self._signals = SIGNALS.copy()
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
                """
                -----------------------------------------------


                        CHECK HERE TO ADD MORE SIGNALS        

                -----------------------------------------------
                """
                if (self.name == "s00"):
                    valid = self.bus.axis_tvalid.value
                    last = self.bus.axis_tlast.value
                    data = self.bus.axis_tdata.value
                    reset = self.bus.axis_aresetn.value
                    start_frame = self.bus.start_frame.value
                    start_capture = self.bus.start_capture.value
                else:
                    addr = self.bus.bram_address.value
                    data = self.bus.bram_data.value
                    wea = self.bus.bram_wea.value

                    valid = wea # valid transactions happen when wea is high
                ready = 1 #bram always ready


                if valid and ready:
                    self.transactions += 1
                    if (self.name == "m00"):
                        thing = dict(data=data,name=self.name,count=self.transactions, addr=addr.integer, wea=wea)
                    else:
                        thing = dict(data=data,name=self.name,count=self.transactions, reset=reset, last=last, valid=valid, start_frame=start_frame, start_capture=start_capture)
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
    self._signals = SIGNALS.copy()
    BusDriver.__init__(self, dut, name, clk)
    self.clock = clk

    """
    -----------------------------------------------


            CHECK HERE TO ADD MORE SIGNALS        

    -----------------------------------------------
    """

    self.bus.start_frame.value = 0
    self.bus.start_capture.value = 0

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

    """
    -----------------------------------------------


            CHECK HERE TO ADD MORE SIGNALS        

    -----------------------------------------------
    """

    if (not self._checker):
        return
    
    rising_edge = RisingEdge(self.clock) # make these coroutines once and reuse
    falling_edge = FallingEdge(self.clock)
    read_only = ReadOnly() #This is
    
    if "type" not in value:
        raise Exception("type missing in value")
    if value["type"] == "single":
        #always ready so no need for awaiting ready
        await falling_edge
        self.bus.axis_tstrb.value = value["contents"]["strb"]
        self.bus.axis_tdata.value = value["contents"]["data"]
        self.bus.axis_tlast.value = value["contents"]["last"]
        self.bus.start_frame.value = value["contents"]["start_frame"]
        self.bus.start_capture.value = value["contents"]["start_capture"]

        self.bus.axis_tvalid.value = 1
        await falling_edge
        self.bus.axis_tvalid.value = 0


    elif value["type"] == "burst":
        temp = np.copy(value["contents"]["data"])
        ind = 0
        #await falling_edge
        while ind < len(temp):
            # always ready so no need for awaiting ready
            await falling_edge
            self.bus.axis_tdata.value = temp[ind].item()
            self.bus.axis_tstrb.value = cocotb.binary.BinaryValue(value = 15, n_bits = self.bus.axis_tstrb.value.n_bits)
            if ((ind + 1)% LINE_LENGTH == 0 and ind > 0):
                # sending the tlast @ end of each line 
                self.bus.axis_tlast.value = 1
            else:
                # single cycle high
                self.bus.axis_tlast.value = 0

            if (ind == 0):
                # this is beginning, so send the start signals
                self.bus.start_frame.value = 1
            else:
                # single cycle high
                self.bus.start_frame.value = 0

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

'''
test structure will be
1. simple single pixel send
2. single pixel send with feathering start frame (and see if stop decode affects it maybe)
3. send random commands and make sure its consistent
4. send a full 640xHEIGHT frame feathering start frame
5. send multiple 640xHEIGHT frames (to make sure it kicks back to idle and waits until decoding is finished to start a new frame)
'''

@cocotb.test()
async def test_simple_send_with_start_frame(dut, *args):
    """simple send"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter


    # feed the driver
    # either single(s)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 1, "start_capture": 1}}
    #send the data
    tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 5) #figure out how many cycles to wait for a result

    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result


@cocotb.test()
async def test_simple_send_with_no_start_frame(dut, *args):
    """checking to make sure bram writes only occur when first start capture is triggered, then start frame is triggered"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter


    # feed the driver
    # either single(s)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 0}}
    tester.input_driver.append(data)
    # data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 1, "start_capture": 0}}
    # tester.input_driver.append(data)
    # data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 0}}
    # tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 20) #figure out how many cycles to wait for a result

    # cant use below asserts cause its not necessarily true that we want input to equal output
    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result

@cocotb.test()
async def test_simple_send_with_delayed_start_frame(dut, *args):
    """checking to make sure bram writes only occur when first start capture is triggered, then start frame is triggered"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter


    # feed the driver
    # either single(s)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 0}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 1, "start_capture": 0}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 0}}
    tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 20) #figure out how many cycles to wait for a result

    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result

@cocotb.test()
async def test_random_inputs(dut, *args):
    """python model and verilog state machine consistency test"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter


    # feed the driver
    # either single(s)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 0, "start_frame": 0, "start_capture": 0}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 0, "start_frame": 1, "start_capture": 0}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 20) #figure out how many cycles to wait for a result

    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result

@cocotb.test()
async def test_full_frame(dut, *args):
    """test full 640xHEIGHT frame send"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter
    # feed the driver
    # either single(s)
    data  = {'type':'burst', "contents":{"data": [4294967295]*(640*240) + [0]*(640*240)}}
    tester.input_driver.append(data)
    
    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 640*HEIGHT + 10) #figure out how many cycles to wait for a result

    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
    raise tester.scoreboard.result


@cocotb.test()
async def test_full_frame_multiple(dut, *args):
    """test full 640xHEIGHT frame multiple send"""
    tester = Tester(dut, verbose=(os.environ.get('-v', False) in TRUE), history=True)
    tester.start()
    cocotb.start_soon(Clock(dut.s00_axis_aclk, 10, units="ns").start())
    # no ready awaited b/c module doesnt have one
    # await set_ready(dut.s00_axis_aclk, dut,1)
    await reset(dut.s00_axis_aclk, dut.s00_axis_aresetn,2,0)
    # make sure model is reset as well
    tester.model(dict(reset=0, count=-1, donttrack=1))# count doesnt matter
    # feed the driver
    # either single(s)
    data  = {'type':'burst', "contents":{"data": [4294967295]*(640*(HEIGHT//2)) + [0]*(640*HEIGHT//2)}}
    tester.input_driver.append(data)
    await ClockCycles(dut.s00_axis_aclk, 640*HEIGHT + 10) #figure out how many cycles to wait for a result
    # check if it kicks back to idle correctly
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 1, "start_capture": 0}}
    tester.input_driver.append(data)
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 1, "start_capture": 0}}
    tester.input_driver.append(data)
    await ClockCycles(dut.s00_axis_aclk, 640*HEIGHT + 10)
    # then send actual next frame
    print("START NEXT FRAME----------------------------------------------------------------------")
    data  = {'type':'single', "contents":{"strb": 15, "last": 0, "data": 4294967295, "start_frame": 0, "start_capture": 1}}
    tester.input_driver.append(data)
    data  = {'type':'burst', "contents":{"data": [4294967295]*(640*HEIGHT//2) + [0]*(640*HEIGHT//2)}}
    tester.input_driver.append(data)

    # wait and apply backpressure
    await ClockCycles(dut.s00_axis_aclk, 640*HEIGHT + 101) #figure out how many cycles to wait for a result

    assert (tester.input_mon.transactions - tester.decrement) ==tester.output_mon.transactions, f"Transaction Count doesn't match! Expected {tester.input_mon.transactions} got {tester.output_mon.transactions}:/"
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
    sources = [proj_path / "hdl" / "frame_capture.sv"] #grow/modify this as needed.
    # i believe you can add another source like below?
    # sources.append(proj_path / "hdl" / "verilog_file.sv")
    build_test_args = ["-Wall"]#,"COCOTB_RESOLVE_X=ZEROS"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="frame_capture",
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True
    )
    runner.test(
        hdl_toplevel="frame_capture",
        test_module="test_frame_capture",
        test_args=run_test_args,
        waves=True
    )
 
if __name__ == "__main__":
    runner()
