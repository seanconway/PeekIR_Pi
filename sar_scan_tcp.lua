-- SAR TCP Client for mmWave Studio
-- Connects to the SAR coordinator and executes radar capture commands.
--
-- REQUIRES: LuaSocket in mmWave Studio's Lua environment.
--   1. Download LuaSocket for Lua 5.1: https://github.com/lunarmodules/luasocket
--   2. Place socket.lua in mmWave Studio's Lua path
--   3. Place socket/core.dll in mmWave Studio's Lua clibs path
--
-- PROTOCOL (newline-delimited text):
--   Server -> Client: ARM <filepath> | CAPTURE | CONFIGURE <frames> <period> | PING | SHUTDOWN
--   Client -> Server: RADAR | ARMED | CAPTURE_DONE | CONFIGURED | PONG | ERROR <msg>

-- =================================================================================
-- CONFIGURATION
-- =================================================================================
local SERVER_HOST = "127.0.0.1"   -- Coordinator IP (localhost if on same Windows machine)
local SERVER_PORT = 5555

local num_frames = 100
local frame_periodicity = 18      -- ms
local capture_buffer_ms = 1000    -- extra wait after frame for DCA1000 flush

-- =================================================================================
-- TCP HELPERS
-- =================================================================================
local ok, socket = pcall(require, "socket")
if not ok then
    WriteToLog("ERROR: LuaSocket not found. Install it for TCP support.\n", "red")
    WriteToLog("See script header for installation instructions.\n", "red")
    return
end

local function tcp_send(tcp, msg)
    tcp:send(msg .. "\n")
    WriteToLog("  TX: " .. msg .. "\n", "black")
end

local function tcp_recv(tcp)
    local line, err = tcp:receive("*l")
    if line then
        WriteToLog("  RX: " .. line .. "\n", "blue")
    end
    return line, err
end

-- =================================================================================
-- SENSOR INITIALIZATION
-- =================================================================================
WriteToLog("=== SAR TCP Client Starting ===\n", "blue")

ar1.StopFrame()
ar1.CaptureCardConfig_StopRecord()
RSTD.Sleep(20)

if (ar1.ProfileConfig(0, 77, 7, 6, 63, 0, 0, 0, 0, 0, 0, 63.343, 0, 512, 9121, 0, 0, 30) == 0) then
    WriteToLog("ProfileConfig Success\n", "green")
else
    WriteToLog("ProfileConfig Failure\n", "red")
end

if (ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0) == 0) then
    WriteToLog("ChirpConfig Success\n", "green")
end

if (ar1.FrameConfig(0, 0, num_frames, 1, frame_periodicity, 0, 512, 1) == 0) then
    WriteToLog("FrameConfig Success\n", "green")
else
    WriteToLog("FrameConfig Failure\n", "red")
end

ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)

WriteToLog("Sensor initialized.\n", "green")

-- Warm up DCA1000 with a dummy capture to flush internal state.
-- The first StartRecord after init often fails to write to disk.
WriteToLog("Warming up DCA1000 (dummy capture)...\n", "blue")
ar1.CaptureCardConfig_StartRecord("C:\\temp\\dca_warmup_dummy.bin", 1)
RSTD.Sleep(50)
ar1.CaptureCardConfig_StopRecord()
RSTD.Sleep(200)
WriteToLog("DCA1000 warmup complete.\n", "green")

-- =================================================================================
-- CONNECT TO COORDINATOR
-- =================================================================================
WriteToLog("Connecting to " .. SERVER_HOST .. ":" .. SERVER_PORT .. "...\n", "blue")

local tcp = socket.tcp()
tcp:settimeout(30)
local conn_ok, conn_err = tcp:connect(SERVER_HOST, SERVER_PORT)
if not conn_ok then
    WriteToLog("Connection failed: " .. tostring(conn_err) .. "\n", "red")
    return
end
tcp:settimeout(nil)

WriteToLog("Connected to coordinator.\n", "green")
tcp_send(tcp, "RADAR")

-- =================================================================================
-- COMMAND LOOP
-- =================================================================================
WriteToLog("Listening for commands...\n", "blue")

while true do
    local cmd, err = tcp_recv(tcp)
    if not cmd then
        WriteToLog("Connection lost: " .. tostring(err) .. "\n", "red")
        break
    end

    if cmd:sub(1, 9) == "CONFIGURE" then
        local nf, fp = cmd:match("CONFIGURE (%d+) (%d+)")
        if nf and fp then
            num_frames = tonumber(nf)
            frame_periodicity = tonumber(fp)
            if (ar1.FrameConfig(0, 0, num_frames, 1, frame_periodicity, 0, 512, 1) == 0) then
                WriteToLog("Reconfigured: " .. num_frames .. " frames @ " .. frame_periodicity .. "ms\n", "green")
                tcp_send(tcp, "CONFIGURED")
            else
                WriteToLog("FrameConfig failed after CONFIGURE\n", "red")
                tcp_send(tcp, "ERROR FrameConfig failed")
            end
        else
            tcp_send(tcp, "ERROR bad CONFIGURE args")
        end

    elseif cmd:sub(1, 3) == "ARM" then
        local filepath = cmd:sub(5)
        WriteToLog("Arming DCA1000: " .. filepath .. "\n", "black")
        if (ar1.CaptureCardConfig_StartRecord(filepath, 1) == 0) then
            WriteToLog("DCA1000 armed.\n", "green")
            tcp_send(tcp, "ARMED")
        else
            WriteToLog("ARM failed!\n", "red")
            tcp_send(tcp, "ERROR ARM failed")
        end

    elseif cmd == "CAPTURE" then
        WriteToLog("Starting frame capture...\n", "blue")
        if (ar1.StartFrame() == 0) then
            WriteToLog("Frame started.\n", "green")
            local wait_ms = (num_frames * frame_periodicity) + capture_buffer_ms
            WriteToLog("Capturing for " .. wait_ms .. "ms...\n", "black")
            RSTD.Sleep(wait_ms)
            ar1.CaptureCardConfig_StopRecord()
            RSTD.Sleep(50)
            WriteToLog("Capture complete.\n", "green")
            tcp_send(tcp, "CAPTURE_DONE")
        else
            WriteToLog("StartFrame failed!\n", "red")
            tcp_send(tcp, "ERROR StartFrame failed")
        end

    elseif cmd == "PING" then
        tcp_send(tcp, "PONG")

    elseif cmd == "SHUTDOWN" then
        WriteToLog("Shutdown received.\n", "blue")
        tcp_send(tcp, "OK")
        break

    else
        WriteToLog("Unknown command: " .. cmd .. "\n", "red")
        tcp_send(tcp, "ERROR unknown command")
    end
end

tcp:close()
WriteToLog("=== SAR TCP Client Disconnected ===\n", "blue")
