import serial
from typing import List, Union


class NKTControl:
    """
    NKTControl Class to control super continuum (SuperK) lasers from NKT.

    This class uses serial communication (virtually through a USB connection) to
    communicate with NKT SuperK Extreme products. All code is tested on a
    SuperK Extreme EXU-6.

    Can obtain and change the power level and upper/lower bandwidths, turn
    emission on or off, and reset the interlock.

    Attributes:
        timeout: Timeout in seconds for the serial port (default: 0.05)
    """

    # Constants
    START_TEL = "0D"
    END_TEL = "0A"
    ADDR_LASER = "0F"
    ADDR_VARIA = "10"
    HOST = "A2"
    MSG_READ = "04"
    MSG_WRITE = "05"

    # CRC lookup table
    CRC_LOOKUP_TABLE = [
        0,
        4129,
        8258,
        12387,
        16516,
        20645,
        24774,
        28903,
        33032,
        37161,
        41290,
        45419,
        49548,
        53677,
        57806,
        61935,
        4657,
        528,
        12915,
        8786,
        21173,
        17044,
        29431,
        25302,
        37689,
        33560,
        45947,
        41818,
        54205,
        50076,
        62463,
        58334,
        9314,
        13379,
        1056,
        5121,
        25830,
        29895,
        17572,
        21637,
        42346,
        46411,
        34088,
        38153,
        58862,
        62927,
        50604,
        54669,
        13907,
        9842,
        5649,
        1584,
        30423,
        26358,
        22165,
        18100,
        46939,
        42874,
        38681,
        34616,
        63455,
        59390,
        55197,
        51132,
        18628,
        22757,
        26758,
        30887,
        2112,
        6241,
        10242,
        14371,
        51660,
        55789,
        59790,
        63919,
        35144,
        39273,
        43274,
        47403,
        23285,
        19156,
        31415,
        27286,
        6769,
        2640,
        14899,
        10770,
        56317,
        52188,
        64447,
        60318,
        39801,
        35672,
        47931,
        43802,
        27814,
        31879,
        19684,
        23749,
        11298,
        15363,
        3168,
        7233,
        60846,
        64911,
        52716,
        56781,
        44330,
        48395,
        36200,
        40265,
        32407,
        28342,
        24277,
        20212,
        15891,
        11826,
        7761,
        3696,
        65439,
        61374,
        57309,
        53244,
        48923,
        44858,
        40793,
        36728,
        37256,
        33193,
        45514,
        41451,
        53516,
        49453,
        61774,
        57711,
        4224,
        161,
        12482,
        8419,
        20484,
        16421,
        28742,
        24679,
        33721,
        37784,
        41979,
        46042,
        49981,
        54044,
        58239,
        62302,
        689,
        4752,
        8947,
        13010,
        16949,
        21012,
        25207,
        29270,
        46570,
        42443,
        38312,
        34185,
        62830,
        58703,
        54572,
        50445,
        13538,
        9411,
        5280,
        1153,
        29798,
        25671,
        21540,
        17413,
        42971,
        47098,
        34713,
        38840,
        59231,
        63358,
        50973,
        55100,
        9939,
        14066,
        1681,
        5808,
        26199,
        30326,
        17941,
        22068,
        55628,
        51565,
        63758,
        59695,
        39368,
        35305,
        47498,
        43435,
        22596,
        18533,
        30726,
        26663,
        6336,
        2273,
        14466,
        10403,
        52093,
        56156,
        60223,
        64286,
        35833,
        39896,
        43963,
        48026,
        19061,
        23124,
        27191,
        31254,
        2801,
        6864,
        10931,
        14994,
        64814,
        60687,
        56684,
        52557,
        48554,
        44427,
        40424,
        36297,
        31782,
        27655,
        23652,
        19525,
        15522,
        11395,
        7392,
        3265,
        61215,
        65342,
        53085,
        57212,
        44955,
        49082,
        36825,
        40952,
        28183,
        32310,
        20053,
        24180,
        11923,
        16050,
        3793,
        7920,
    ]

    def __init__(self, port: str = "COM6"):
        self.timeout = 0.05  # Timeout in seconds for the serial port
        self.serial_conn = None  # Serial communication object
        self.port = port

    def connect(self) -> None:
        """
        Connects to the laser.

        Detects which serial ports are available and tries to connect
        to each in turn until the laser is connected. Always first
        method to be called.

        See also: disconnect
        """
        # For this implementation, we'll use a specific port as in the MATLAB code
        # In a real implementation, you might want to scan available ports

        try:
            self.serial_conn = serial.Serial(
                port=self.port, baudrate=115200, timeout=self.timeout
            )

            data = "30"
            self._send_telegram(self.ADDR_LASER, self.MSG_READ, data)
            out = self._get_telegram(9)

            if not out:
                self.serial_conn.close()
                self.serial_conn = None
            elif out[0] == self.START_TEL:
                print("Laser connected")
            else:
                self.serial_conn.close()
                self.serial_conn = None

        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.serial_conn = None

    def disconnect(self) -> None:
        """
        Close serial port connection thus disconnecting laser.

        See also: connect
        """
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Laser disconnected")

    def set_timeout(self, timeout: float) -> None:
        """
        Sets the timeout for the serial port in seconds.

        This function may be useful to play around with if either
        communication fails or faster updates are required.

        Args:
            timeout: Timeout value given in seconds.

        See also: get_timeout
        """
        self.timeout = timeout
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.timeout = timeout

    def get_timeout(self) -> float:
        """
        Obtains the timeout for the serial port in seconds.

        Returns:
            Current timeout value given in seconds.

        See also: set_timeout
        """
        return self.timeout

    def get_status(self) -> List[str]:
        """
        Obtains the status of the laser.

        Checks whether the serial port is open, whether emission is on
        or off, and whether the interlock needs resetting. Also gives
        a notification when the clock battery is low.

        Returns:
            Status given as [emission, interlock, battery].
            Emission 0/1 is off/on;
            Interlock 0/1 is on/off;
            Battery 0/1 is okay/low.
        """
        data = "66"
        self._send_telegram(self.ADDR_LASER, self.MSG_READ, data)
        out = self._get_telegram(10)

        hexdata = out[6] + out[5]  # Note: Python is 0-indexed
        decdata = int(hexdata, 16)
        status = bin(decdata)[2:].zfill(16)  # Convert to 16-bit binary string

        output = [status[15], status[14], status[8]]  # Python strings are 0-indexed

        print(f"Com port {'open' if self.serial_conn.is_open else 'closed'}")
        print("Emission on" if output[0] == "1" else "Emission off")
        print("Interlock on" if output[1] == "0" else "Interlock needs resetting")
        if output[2] == "1":
            print("Clock battery voltage low")

        return output

    def get_varia_status(self) -> List[str]:
        """
        Obtains the status of the Varia filter.

        Checks whether each shutter is open or closed and whether any
        of the filters are moving.

        Returns:
            Given as [Shutter 1, Shutter 2, Filter 1, Filter 2, Filter 3]
            Shutters 0/1 is closed/open;
            filters 0/1 is stationary/moving.
        """
        data = "66"
        self._send_telegram(self.ADDR_VARIA, self.MSG_READ, data)
        out = self._get_telegram(10)

        hexdata = out[6] + out[5]  # Note: Python is 0-indexed
        decdata = int(hexdata, 16)
        status = bin(decdata)[2:].zfill(16)  # Convert to 16-bit binary string

        output = [
            status[7],
            status[6],
            status[3],
            status[2],
            status[1],
        ]  # Python strings are 0-indexed

        print("Shutter 1 open" if output[0] == "1" else "Shutter 1 closed")
        print("Shutter 2 open" if output[1] == "1" else "Shutter 2 closed")
        if output[2] == "1":
            print("Filter 1 moving")
        if output[3] == "1":
            print("Filter 2 moving")
        if output[4] == "1":
            print("Filter 3 moving")

        return output

    def reset_interlock(self) -> None:
        """Resets the interlock circuit."""
        data = ["32", "01"]
        self._send_telegram(self.ADDR_LASER, self.MSG_WRITE, data)
        self._get_telegram(8)

    def emission_on(self) -> None:
        """Turns emission on."""
        data = ["30", "03"]
        self._send_telegram(self.ADDR_LASER, self.MSG_WRITE, data)
        self._get_telegram(8)

    def emission_off(self) -> None:
        """Turns emission off."""
        data = ["30", "00"]
        self._send_telegram(self.ADDR_LASER, self.MSG_WRITE, data)
        self._get_telegram(8)

    def get_power_level(self) -> float:
        """
        Obtain power level.

        Returns:
            Power level given in percent with 0.1% precision.

        See also: set_power_level
        """
        data = "37"
        self._send_telegram(self.ADDR_LASER, self.MSG_READ, data)
        out = self._get_telegram(10)

        # Obtain hex power level from telegram
        hex_power_level = out[6] + out[5]
        return 0.1 * int(hex_power_level, 16)  # Convert to percent

    def set_power_level(self, power_level: float) -> None:
        """
        Set power level.

        Args:
            power_level: Desired power level given in percentage with 0.1% precision.

        See also: get_power_level
        """
        power_level_2 = int(10 * power_level)  # Convert to 0.1%
        hex_power_level = format(power_level_2, "04X")  # Convert to 4-digit hex

        data = ["37", hex_power_level[2:4], hex_power_level[0:2]]
        self._send_telegram(self.ADDR_LASER, self.MSG_WRITE, data)
        self._get_telegram(8)

    def get_lower_bandwidth(self) -> float:
        """
        Get current lower bandwidth setting of Varia.

        Returns:
            Lower band edge of the current filter setting of the Varia filter
            given in nm specified with 0.1nm precision.

        See also: set_lower_bandwidth, get_upper_bandwidth, set_upper_bandwidth
        """
        data = "34"
        self._send_telegram(self.ADDR_VARIA, self.MSG_READ, data)
        out = self._get_telegram(10)

        hex_lower_bw = out[6] + out[5]
        return 0.1 * int(hex_lower_bw, 16)

    def set_lower_bandwidth(self, lower_bandwidth: float) -> None:
        """
        Set desired lower bandwidth setting of Varia.

        Args:
            lower_bandwidth: Lower band edge to be set on the Varia filter
                            given in nm specified with 0.1nm precision.

        See also: get_lower_bandwidth, get_upper_bandwidth, set_upper_bandwidth
        """
        lower_bw_2 = int(10 * lower_bandwidth)  # units 0.1nm
        hex_lower_bw = format(lower_bw_2, "04X")

        data = ["34", hex_lower_bw[2:4], hex_lower_bw[0:2]]
        self._send_telegram(self.ADDR_VARIA, self.MSG_WRITE, data)
        self._get_telegram(8)

    def get_upper_bandwidth(self) -> float:
        """
        Get current upper bandwidth setting of Varia.

        Returns:
            Upper band edge of the current filter setting of the Varia filter
            given in nm specified with 0.1nm precision.

        See also: set_lower_bandwidth, get_lower_bandwidth, set_upper_bandwidth
        """
        data = "33"
        self._send_telegram(self.ADDR_VARIA, self.MSG_READ, data)
        out = self._get_telegram(10)

        hex_upper_bw = out[6] + out[5]
        return 0.1 * int(hex_upper_bw, 16)

    def set_upper_bandwidth(self, upper_bandwidth: float) -> None:
        """
        Set desired upper bandwidth setting of Varia.

        Args:
            upper_bandwidth: Upper band edge to be set on the Varia filter
                            given in nm specified with 0.1nm precision.

        See also: get_lower_bandwidth, set_lower_bandwidth, get_upper_bandwidth
        """
        upper_bw_2 = int(10 * upper_bandwidth)  # units 0.1nm
        hex_upper_bw = format(upper_bw_2, "04X")

        data = ["33", hex_upper_bw[2:4], hex_upper_bw[0:2]]
        self._send_telegram(self.ADDR_VARIA, self.MSG_WRITE, data)
        self._get_telegram(8)

    def _send_telegram(
        self, address: str, msg_type: str, data: Union[str, List[str]]
    ) -> None:
        """
        Sends a telegram to the laser.

        Info on communication protocol to be found in documentation
        for SuperK laser.

        Args:
            address: Address of the laser (16 bit)
            msg_type: Type of message (16 bit)
            data: Data - if applicable (can be string or list of strings)

        See also: _get_telegram
        """
        if isinstance(data, str):
            data = [data]

        message = [address, self.HOST, msg_type] + data
        crc = self._crc_value(message)
        crc1 = crc[0:2]
        crc2 = crc[2:4]

        t = [self.START_TEL] + message + [crc1, crc2, self.END_TEL]

        # Replace special characters
        out = []
        out.append(self.START_TEL)

        for item in t[1:-1]:  # Skip first (START_TEL) and last (END_TEL)
            if item == "0A":
                out.extend(["5E", "4A"])
            elif item == "0D":
                out.extend(["5E", "4D"])
            elif item == "5E":
                out.extend(["5E", "9E"])
            else:
                out.append(item)

        out.append(self.END_TEL)

        # Convert to bytes and send
        byte_data = bytes([int(x, 16) for x in out])
        self.serial_conn.write(byte_data)

    def _get_telegram(self, size: int) -> List[str]:
        """
        Receives a telegram from the laser.

        Info on communication protocol to be found in documentation
        for SuperK laser.

        Args:
            size: The expected length of the received telegram.

        Returns:
            List of hex strings representing the received telegram.

        See also: _send_telegram
        """
        received = self.serial_conn.read(size)
        if not received:
            return []

        # Convert to hex strings
        t = [format(x, "02X") for x in received]

        # Replace any special characters
        out = []
        i = 0
        while i < len(t):
            if t[i] == "5E":
                if i + 1 < len(t):
                    if t[i + 1] == "4A":
                        out.append("0A")
                    elif t[i + 1] == "4D":
                        out.append("0D")
                    elif t[i + 1] == "9E":
                        out.append("5E")
                    i += 2
                else:
                    i += 1
            else:
                out.append(t[i])
                i += 1

        # Check if transmission is complete and receive any additional bytes if necessary
        while out and out[-1] != self.END_TEL:
            additional = self.serial_conn.read(1)
            if not additional:
                break
            out.append(format(ord(additional), "02X"))

        return out

    def _crc_value(self, message: List[str]) -> str:
        """
        Finds the CRC value of a given message.

        Info on communication protocol to be found in documentation
        for SuperK laser. CRC value found through look-up table.

        Args:
            message: Data for which CRC value needs to be found.

        Returns:
            The corresponding CRC value as a 4-character hex string.
        """
        crc = 0x0000

        for item in message:
            byte = int(item, 16)
            index = (byte ^ (crc >> 8)) & 0xFF
            crc = self.CRC_LOOKUP_TABLE[index] ^ ((crc << 8) & 0xFFFF)

        return format(crc, "04X")
