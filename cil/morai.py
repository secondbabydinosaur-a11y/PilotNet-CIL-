### 모라이 카메라, 차량 상태 수신, 그리고 제어 명령 송신하는 파일임. ###

###무슨 통신 써서 모라이랑 통신할건지 먼저 정해야함...###

#1. 컴퓨터 네트워크에서 한번에 보낼 수 있는 데이터 최대 크기는 일반적으로 1500bytes. 
# 하지만 카메라로 찍은 jpeg 이미지 한장은 아무리 작아도 수십 수백 kb임. 그래서 이 데이터를 한번에 udp 패킷으로 던지면 유실됨...
# 그래서 모라이에서 jpeg 이미지를 여러개의 조각으로 쪼개고 각 조각에 "나는 몇번째 조각이고 총 몇개다"라는 헤더 정보를 붙여서 순서대로 전송함. 
# 이후 수신(지금 이 파일)에서는 들어오는 조각들을 모았다가 모든 조각이 다 도착하면 하나의 jpeg 이미지로 다시 합쳐서 디코딩함.

# 차량 제어에서는 여전히 can 통신을 주력으로...(실시간성 보장, 노이즈 강하고 비용 절감 효과)
# 하지만 요즘 카메라나 라이다같은 센서들은 용량이 매우커서 UDP/TCP(ip통신)을 적극적으로 활용함...!!!




################ 앞 내용 꼭 읽어보기!!! #####################





import argparse  # 터미널 실행 옵션을 읽는다.
import json  # config.json을 읽는다.
import math  # NaN과 무한대 값을 검사한다.
import selectors  # 여러 UDP 소켓을 함께 감시한다.
import socket  # UDP 통신을 사용한다.
import struct  # 바이너리 패킷을 숫자로 변환하거나 생성한다.
import time  # 수신 시각과 경과 시간을 측정한다.
from collections import Counter, OrderedDict, deque  # 통계와 조립 상태를 보관한다.
from pathlib import Path  # 파일 경로를 계산한다.


BASE = Path(__file__).resolve().parent  # 이 파일이 들어 있는 cil 폴더다.


def load_config(path=BASE / "config.json"):  # 설정 파일을 읽는다.
    with Path(path).open(encoding="utf-8") as stream:  # UTF-8로 JSON을 연다.
        return json.load(stream)  # 설정을 딕셔너리로 반환한다.


def sensor_stamp(sec, nsec):  # 초·나노초를 하나의 정수 시각으로 합친다.
    if not 0 <= nsec < 1_000_000_000:  # 나노초 부분의 범위를 검사한다.
        raise ValueError("invalid sensor nanoseconds")  # 잘못된 시각을 거부한다.

    return sec * 1_000_000_000 + nsec  # 부동소수점 오차 없이 ns로 반환한다.


def parse_ego(packet):  # 일반 Ego Vehicle Status 패킷을 해석한다.
    if len(packet) not in (181, 229) or packet[:11] != b"#MoraiInfo$":  # 종류·길이다.
        raise ValueError("expected ordinary Ego Vehicle Status")  # 다른 형식은 거부한다.

    size = struct.unpack_from("<I", packet, 11)[0]  # 본문 길이를 uint32로 읽는다.

    if size != len(packet) - 29 or packet[-2:] != b"\r\n":  # 길이와 끝을 검사한다.
        raise ValueError("invalid Ego packet size or tail")  # 손상된 패킷을 거부한다.

    payload = packet[27:-2]  # 헤더와 종료 문자를 제외한 본문이다.
    sec, nsec = struct.unpack_from("<II", payload, 0)  # 센서 시각을 읽는다.
    ctrl_mode, gear = struct.unpack_from("<bb", payload, 8)  # 제어 모드·기어다.

    offsets = {  # 각 float32 값이 본문에서 시작하는 바이트 위치다.
        "speed_kph": 10,  # 차량 속도다.
        "accel": 18,  # 상태 패킷의 가속 페달 값이다.
        "brake": 22,  # 상태 패킷의 브레이크 값이다.
        "x_m": 50,  # 지도 좌표계의 X 위치다.
        "y_m": 54,  # 지도 좌표계의 Y 위치다.
        "z_m": 58,  # 지도 좌표계의 Z 위치다.
        "roll_deg": 62,  # 차량 Roll이다.
        "pitch_deg": 66,  # 차량 Pitch다.
        "yaw_deg": 70,  # 차량 헤딩이다.
        "vx_kph": 74,  # 속도 벡터 X 성분이다.
        "vy_kph": 78,  # 속도 벡터 Y 성분이다.
        "vz_kph": 82,  # 속도 벡터 Z 성분이다.
        "yaw_rate_deg_s": 94,  # 초당 헤딩 변화량이다.
        "steer_raw_deg": 110,  # 상태 패킷의 실제 조향 값이다.
    }

    values = {  # 지정한 위치에서 float32 값을 하나씩 추출한다.
        key: struct.unpack_from("<f", payload, pos)[0]  # <f는 little-endian float32다.
        for key, pos in offsets.items()  # 항목 이름과 바이트 위치를 순회한다.
    }

    if not all(math.isfinite(value) for value in values.values()):  # 실수를 검사한다.
        raise ValueError("non-finite Ego value")  # NaN이나 무한대가 있으면 거부한다.

    if abs(values["speed_kph"]) > 400 or abs(values["steer_raw_deg"]) > 90:  # 범위다.
        raise ValueError("implausible Ego speed or steering")  # 해석 오류를 의심한다.

    values["stamp_ns"] = sensor_stamp(sec, nsec)  # 이미지와 매칭할 센서 시각이다.
    values["ctrl_mode"] = ctrl_mode  # 현재 제어 모드를 보관한다.
    values["gear"] = gear  # 현재 기어를 보관한다.
    values["link_id"] = (  # 차량 상태 패킷에 담긴 지도 링크 ID를 읽는다.
        payload[114:152]  # 링크 ID가 들어 있는 바이트 범위다.
        .split(b"\x00", 1)[0]  # 문자열 뒤의 빈 패딩을 제거한다.
        .decode("ascii", errors="replace")  # 바이트를 문자열로 변환한다.
    )

    return values  # 해석된 차량 상태를 반환한다.


class CameraAssembler:  # UDP 조각을 JPEG 한 장으로 조립한다.
    def __init__(self, max_pending=6):  # 미완성 이미지를 최대 6장까지 보관한다.
        if max_pending < 1:  # 보관 개수의 범위를 검사한다.
            raise ValueError("max_pending must be positive")  # 잘못된 설정을 거부한다.

        self.max_pending = max_pending  # 미완성 이미지 보관 상한이다.
        self.pending = OrderedDict()  # 센서 시각별로 수신한 조각을 보관한다.
        self.completed = deque(maxlen=128)  # 이미 완성한 이미지의 시각을 기억한다.
        self.evicted = 0  # 미완성 상태로 폐기한 이미지 수다.

    def push(self, packet):  # 새 UDP 조각 하나를 추가한다.
        if len(packet) < 19 or packet[:3] != b"MOR":  # 카메라 헤더를 검사한다.
            raise ValueError("invalid camera header")  # 다른 패킷을 거부한다.

        sec, nsec, index, length = struct.unpack_from("<IIII", packet, 3)  # 조각 정보다.
        stamp = sensor_stamp(sec, nsec)  # 같은 이미지의 조각을 묶을 기준이다.

        if index >= 512 or not 1 <= length <= 65488 or 19 + length > len(packet):
            raise ValueError("invalid camera fragment")  # 잘못된 번호·길이를 거부한다.

        if stamp in self.completed:  # 이미 반환한 이미지의 조각이라면
            return None  # 동일한 이미지를 다시 반환하지 않는다.

        if stamp not in self.pending:  # 처음 보는 이미지 시각이라면
            if len(self.pending) >= self.max_pending:  # 미완성 이미지가 너무 많으면
                self.pending.popitem(last=False)  # 가장 먼저 등록된 이미지를 버린다.
                self.evicted += 1  # 폐기 통계에 반영한다.

            self.pending[stamp] = {}  # 새 이미지의 조각 저장소를 만든다.

        chunks = self.pending[stamp]  # 이번 이미지의 조각 저장소다.
        fragment = packet[19:19 + length]  # 헤더·패딩을 제외한 JPEG 조각이다.

        if index in chunks:  # 같은 번호의 조각을 이미 받았다면
            if chunks[index] != fragment:  # 내용이 다르면 정상 프레임으로 볼 수 없다.
                del self.pending[stamp]  # 해당 이미지의 조립 상태를 폐기한다.
                raise ValueError("conflicting camera fragment")  # 오류로 알린다.

            return None  # 내용이 같은 중복 조각은 무시한다.

        chunks[index] = fragment  # 도착 순서가 아닌 조각 번호로 저장한다.

        if 0 not in chunks or not chunks[0].startswith(b"\xff\xd8"):  # JPEG 시작이다.
            return None  # 시작 조각을 더 기다린다.

        last = max(chunks)  # 현재 받은 조각 중 가장 큰 번호다.

        if not chunks[last].endswith(b"\xff\xd9") or len(chunks) != last + 1:
            return None  # 끝 조각이나 중간에 누락된 조각을 더 기다린다.

        jpeg = b"".join(chunks[i] for i in range(last + 1))  # 번호 순서대로 합친다.
        del self.pending[stamp]  # 완성했으므로 임시 저장소에서 제거한다.
        self.completed.append(stamp)  # 이 이미지는 이미 처리했다고 기록한다.

        return stamp, jpeg  # 센서 시각과 완성된 JPEG 바이트를 반환한다.
    

def encode_control(steer, throttle, brake):  # 제어값을 MORAI 패킷으로 변환한다.
    if not all(math.isfinite(v) for v in (steer, throttle, brake)):  # 값을 검사한다.
        raise ValueError("non-finite control")  # NaN과 무한대를 거부한다.

    if not (-1 <= steer <= 1 and 0 <= throttle <= 1 and 0 <= brake <= 1):
        raise ValueError("control outside MORAI ranges")  # 허용 범위를 벗어나면 거부한다.

    if throttle > 0 and brake > 0:  # 두 페달을 동시에 누르는지 검사한다.
        raise ValueError("simultaneous throttle and brake")  # 페달 선택은 제어기가 한다.

    payload = struct.pack(  # MORAI의 제어 본문을 만든다.
        "<BBB5f",  # unsigned byte 3개와 float32 5개를 넣는다.
        2,  # Auto 제어 모드다.
        4,  # D 기어다.
        1,  # 가속·브레이크 페달로 종방향 제어한다.
        0.0,  # 속도 지정 필드: 페달 제어에서는 사용하지 않는다.
        0.0,  # 가속도 지정 필드: 페달 제어에서는 사용하지 않는다.
        throttle,  # 가속 페달 값: 0~1이다.
        brake,  # 브레이크 값: 0~1이다.
        steer,  # 정규화 조향 명령: -1~1이다.
    )

    return (  # 헤더·본문 길이·예약 영역·본문·종료 문자를 연결한다.
        b"#MoraiCtrlCmd$"  # 제어 패킷 헤더다.
        + struct.pack("<I", len(payload))  # 본문 길이를 넣는다.
        + bytes(12)  # 예약 영역을 0으로 채운다.
        + payload  # 제어 본문을 넣는다.
        + b"\r\n"  # 패킷 종료 문자다.
    )


class MoraiClient:  # 센서 수신과 제어 송신을 묶는 인터페이스다.
    def __init__(self, config):  # 설정 딕셔너리를 받는다.
        self.config = config  # 송신 주소 등에 사용할 설정이다.
        self.selector = selectors.DefaultSelector()  # 수신 가능한 소켓을 찾는다.
        self.receivers = []  # 종료할 때 닫을 수신 소켓 목록이다.
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 송신 소켓이다.

        self.assemblers = {  # 카메라마다 독립적인 이미지 조립기를 만든다.
            name: CameraAssembler()  # 서로 다른 카메라의 조각이 섞이지 않게 한다.
            for name in config["cameras"]  # left·center·right를 순회한다.
        }

        self.counts = Counter()  # 정상 차량 상태 수와 완성 이미지 수를 센다.
        self.errors = Counter()  # 센서별 패킷 해석 오류를 센다.
        self.latest_ego = None  # 아직 차량 상태를 받지 않은 상태다.
        self.latest_ego_recv_ns = None  # 마지막 정상 차량 상태의 PC 수신 시각이다.

        streams = {  # 각 카메라의 수신 포트를 가져온다.
            name: cam["destination_port"]  # 우리 프로그램이 bind할 포트다.
            for name, cam in config["cameras"].items()  # 카메라 설정을 순회한다.
        }
        streams["ego"] = config["ego"]["destination_port"]  # 차량 상태 포트도 추가한다.

        try:  # 초기화 중 실패해도 이미 연 소켓을 정리한다.
            if len(set(streams.values())) != len(streams):  # 수신 포트 중복을 검사한다.
                raise ValueError("destination ports must be distinct")  # 중복을 거부한다.

            for name, port in streams.items():  # 카메라와 Ego 소켓을 각각 만든다.
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP 소켓이다.
                self.receivers.append(sock)  # bind 실패 시에도 정리할 수 있게 등록한다.
                sock.setsockopt(  # 운영체제에 수신 버퍼 크기를 요청한다.
                    socket.SOL_SOCKET,  # 소켓 수준의 설정이다.
                    socket.SO_RCVBUF,  # UDP 수신 버퍼 옵션이다.
                    4 * 1024 * 1024,  # 4MiB를 요청하며 실제 크기는 OS 제한을 따른다.
                )
                sock.bind((config["network"]["bind_ip"], port))  # 수신 포트를 연다.
                sock.setblocking(False)  # 큐가 비면 recv가 기다리지 않게 한다.
                self.selector.register(sock, selectors.EVENT_READ, name)  # 센서명이다.

        except Exception:  # 포트 충돌 등으로 초기화에 실패했다면
            self.close()  # 이미 연 소켓을 정리한다.
            raise  # 원래 오류를 호출 측에 전달한다.

    def poll(self, timeout_s=0.02):  # 도착한 데이터를 이벤트 목록으로 반환한다.
        events = []  # 이번 호출에서 받은 차량 상태와 완성 이미지를 모은다.

        for key, _ in self.selector.select(timeout_s):  # 데이터가 있는 소켓을 찾는다.
            name = key.data  # left·center·right·ego 중 하나다.

            for _ in range(64):  # 한 센서가 다른 센서 처리를 계속 막지 않게 제한한다.
                try:  # 소켓 수신 큐에서 패킷을 꺼낸다.
                    packet = key.fileobj.recv(65535)  # UDP 패킷 하나를 받는다.
                except BlockingIOError:  # 해당 소켓에 남은 패킷이 없으면
                    break  # 다음 센서로 넘어간다.

                recv_ns = time.monotonic_ns()  # PC가 패킷을 읽은 시각이다.

                try:  # 손상된 패킷은 기록하고 다음 패킷을 처리한다.
                    if name == "ego":  # 차량 상태 패킷이라면
                        data = parse_ego(packet)  # 상태 딕셔너리로 변환한다.
                        self.latest_ego = data  # 최신 차량 상태를 갱신한다.
                        self.latest_ego_recv_ns = recv_ns  # 수신 시각도 갱신한다.

                        event = {  # 차량 상태 이벤트를 만든다.
                            "kind": "ego",  # 이벤트 종류다.
                            "stamp_ns": data["stamp_ns"],  # 센서 시각이다.
                            "recv_ns": recv_ns,  # PC 수신 시각이다.
                            "data": data,  # 속도·위치·조향 등이다.
                        }

                    else:  # 카메라 패킷이라면
                        frame = self.assemblers[name].push(packet)  # 조각을 추가한다.

                        if frame is None:  # 이미지가 아직 완성되지 않았다면
                            continue  # 다음 조각을 받는다.

                        stamp, jpeg = frame  # 완성 이미지의 시각과 JPEG다.

                        event = {  # 카메라 이벤트를 만든다.
                            "kind": "camera",  # 이벤트 종류다.
                            "camera": name,  # left·center·right 중 하나다.
                            "stamp_ns": stamp,  # 이미지의 센서 시각이다.
                            "recv_ns": recv_ns,  # 조립 완료 조각의 PC 수신 시각이다.
                            "jpeg": jpeg,  # 원본 JPEG 바이트다.
                        }

                except ValueError:  # 패킷 검사나 해석에 실패했다면
                    self.errors[name] += 1  # 해당 센서의 오류 수를 증가시킨다.
                    continue  # 다른 패킷 처리는 계속한다.

                self.counts[name] += 1  # 카메라는 조각 수가 아닌 완성 이미지 수다.
                events.append(event)  # 수집 코드에 전달할 데이터를 추가한다.

        return events  # 수신한 데이터가 없으면 빈 목록을 반환한다.

    def send_control(self, steer, throttle, brake):  # 호출받은 제어값을 송신한다.
        packet = encode_control(steer, throttle, brake)  # 제어 패킷을 생성한다.
        address = (  # 제어 명령을 받을 MORAI 주소다.
            self.config["network"]["morai_ip"],  # MORAI가 실행 중인 PC의 IP다.
            self.config["control"]["host_port"],  # 제어는 MORAI의 Host Port로 보낸다.
        )
        self.sender.sendto(packet, address)  # 제어 명령을 송신한다.

        return time.monotonic_ns()  # 명령 로그에 사용할 송신 직후 PC 시각이다.

    def close(self):  # 사용한 통신 자원을 정리한다.
        self.selector.close()  # 소켓 감시를 종료한다.

        for sock in self.receivers:  # 열었던 수신 소켓마다
            sock.close()  # 수신 포트를 반환한다.

        self.sender.close()  # 송신 소켓도 닫는다.

    def __enter__(self):  # with MoraiClient(...) 구문을 지원한다.
        return self  # 생성한 통신 객체를 전달한다.

    def __exit__(self, exc_type, exc_value, traceback):  # with 종료 시 호출된다.
        self.close()  # 정상 종료나 예외 발생 시 소켓을 정리한다.



def main():  # 이 파일을 직접 실행하면 수신 상태만 점검한다.
    parser = argparse.ArgumentParser()  # 실행 옵션 해석기를 만든다.
    parser.add_argument(  # 설정 파일 경로를 지정할 수 있게 한다.
        "--config", type=Path, default=BASE / "config.json"  # 기본은 같은 폴더다.
    )
    parser.add_argument(  # 점검 시간을 지정할 수 있게 한다.
        "--seconds", type=float, default=30.0  # 기본 점검 시간은 30초다.
    )
    args = parser.parse_args()  # 터미널에서 전달한 옵션을 읽는다.

    if not math.isfinite(args.seconds) or args.seconds <= 0:  # 시간을 검사한다.
        parser.error("--seconds must be finite and positive")  # 잘못된 옵션이다.

    config = load_config(args.config)  # 설정을 읽는다.

    try:  # Ctrl+C 종료도 처리한다.
        with MoraiClient(config) as client:  # 포트를 열고 종료 시 자동으로 닫는다.
            started = last_report = time.monotonic()  # 시작·마지막 출력 시각이다.
            previous = Counter()  # 이전 출력 때까지의 수신 개수다.

            print("Receive-only check: no control packets are sent.")  # 수신만 한다.

            while time.monotonic() - started < args.seconds:  # 지정한 시간 동안 반복한다.
                client.poll()  # 패킷을 수신하고 상태·통계를 갱신한다.
                now = time.monotonic()  # 현재 PC 시각이다.

                if now - last_report >= 1.0:  # 약 1초마다 출력한다.
                    dt = now - last_report  # 실제 출력 간격이다.
                    rates = " ".join(  # 센서별 실측 수신률을 문자열로 만든다.
                        f"{name}={(client.counts[name] - previous[name]) / dt:.1f}Hz"
                        for name in ("left", "center", "right", "ego")  # 센서 목록이다.
                    )
                    ego = client.latest_ego  # 최근 정상 차량 상태다.

                    if ego is None:  # 차량 상태를 한 번도 받지 못했다면
                        status = "waiting_ego"  # 대기 중이라고 표시한다.
                    else:  # 차량 상태를 받은 적이 있다면
                        age_s = (  # 마지막 정상 상태 수신 이후 경과 시간이다.
                            time.monotonic_ns() - client.latest_ego_recv_ns
                        ) / 1e9  # ns를 초로 변환한다.
                        status = (  # 속도와 상태의 경과 시간을 함께 표시한다.
                            f"speed={ego['speed_kph']:.1f}kph "
                            f"ego_age={age_s:.3f}s"
                        )

                    dropped = sum(  # 미완성으로 폐기된 이미지 수를 합친다.
                        assembler.evicted  # 카메라별 폐기 수다.
                        for assembler in client.assemblers.values()  # 조립기 목록이다.
                    )
                    errors = sum(client.errors.values())  # 패킷 해석 오류의 합이다.

                    print(  # 수신률·차량 상태·오류·미완성 폐기 수를 출력한다.
                        f"{rates} | {status} | errors={errors} incomplete={dropped}"
                    )
                    previous = client.counts.copy()  # 다음 수신률 계산 기준을 갱신한다.
                    last_report = now  # 마지막 출력 시각을 갱신한다.

            print("Finished:", dict(client.counts))  # 전체 수신 개수를 출력한다.

    except KeyboardInterrupt:  # 사용자가 Ctrl+C를 눌렀다면
        print("Receive check stopped.")  # 점검 종료를 알린다.


if __name__ == "__main__":  # 다른 파일에서 import할 때는 실행하지 않는다.
    main()  # 직접 실행했을 때만 수신 점검을 시작한다.    