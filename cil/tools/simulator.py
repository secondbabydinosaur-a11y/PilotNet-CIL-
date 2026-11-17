"""로컬 MORAI의 초기 배치·날씨 설정. 실제 주행 제어는 collect.py의 UDP가 담당한다."""
from pathlib import Path  # 공식 SDK descriptor의 경로를 찾는다.
import socket  # 초기 배치 동안 정지 명령을 보낸다.
import time  # 차량이 노면에 정착할 시간을 준다.
import grpc  # MORAI의 로컬 gRPC API를 호출한다.
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # SDK 메시지를 읽는다.
from google.protobuf.json_format import MessageToDict  # 응답을 딕셔너리로 바꾼다.


class Simulator:  # 수집 시작 전 시뮬레이터 설정만 담당한다.
    def __init__(self, target="127.0.0.1:7789"):  # 같은 PC의 MORAI에 접속한다.
        files = descriptor_pb2.FileDescriptorSet()  # 공식 메시지 정의를 읽을 객체다.
        files.ParseFromString((Path(__file__).resolve().parents[1] / "assets/morai_api.desc").read_bytes())  # SDK 정의다.
        self.pool = descriptor_pool.DescriptorPool()  # 메시지 정의를 보관한다.
        for file in files.file:  # 의존 순서대로 각 정의를 추가한다.
            self.pool.Add(file)  # 직접 패킷 형식을 추측하지 않는다.
        self.channel = grpc.insecure_channel(target)  # 로컬 채널을 연다.
        self.actor = self.msg("morai_sim_api.ObjectInfo", id={"value": "Ego"}, object_type=1, client_key="kcity_test")  # 대상 차량이다.

    def msg(self, name, **values):  # SDK 형식의 메시지를 생성한다.
        return message_factory.GetMessageClass(self.pool.FindMessageTypeByName(name))(**values)  # 값을 채운다.

    def call(self, service, method, request):  # 로컬 gRPC 메서드를 호출한다.
        desc = self.pool.FindServiceByName("morai_sim_api." + service).methods_by_name[method]  # 응답 정의다.
        cls = message_factory.GetMessageClass(desc.output_type)  # 응답 클래스다.
        rpc = self.channel.unary_unary(  # 한 요청에 한 응답을 받는다.
            "/morai_sim_api.{}/{}".format(service, method),  # SDK 메서드 경로다.
            request_serializer=lambda message: message.SerializeToString(),  # 요청을 직렬화한다.
            response_deserializer=cls.FromString,  # 응답을 해석한다.
        )
        result = rpc(request, timeout=3)  # 응답 대기 시간을 제한한다.
        if hasattr(result, "status") and result.status != 1:  # 실패를 성공으로 처리하지 않는다.
            raise RuntimeError("{}: {}".format(method, result))  # 오류 원문을 전달한다.
        return result  # 정상 응답이다.

    def actor_call(self, method, message=None, **values):  # Ego 차량 관련 호출이다.
        request = self.actor if message is None else self.msg("morai_sim_api.actor." + message, actor_info=self.actor, **values)  # 요청이다.
        return self.call("actor.Actor", method, request)  # 차량 API를 호출한다.

    def state(self):  # 실제 위치와 상태를 읽는다.
        return MessageToDict(self.actor_call("GetActorState"), preserving_proto_field_name=True)  # 원래 필드 이름을 유지한다.

    def setup(self, info, config, offset_m=0.0):  # 수집 경로 시작점으로 차량을 배치한다.
        import math  # 횡방향 초기 위치 변화를 계산한다.
        from morai import encode_control  # 검증한 UDP 제어 패킷을 사용한다.
        self.call("simulation.Simulation", "Resume", self.msg("morai_sim_api.Empty"))  # 시뮬레이션을 진행한다.
        self.call("environment.Environment", "SetWeather", self.msg("morai_sim_api.environment.Weather", type=0, instantly=True))  # 맑은 날이다.
        self.call("environment.Environment", "SetTime", self.msg("morai_sim_api.environment.SimulationTime", hour=12, instantly=True))  # 낮 12시다.
        self.actor_call("SetVehicleControlMode", "VehicleControlModeParam", mode=3)  # AV 외부 제어를 선택한다.
        self.actor_call("SetVehicleDynamicsSpeedLimit", "VehicleDynamicsSpeedLimit", speed_limiter=True, speed_limit=float(config["collection"]["speed_cap_kph"]), reset=False)  # 실험 속도 상한이다.
        self.actor_call("SetVelocity", "SetVelocityParam", velocity=0.0)  # 재배치 전 속도를 없앤다.
        x, y, z = info["start_xyz_m"]  # 경로의 지도 시작 좌표다.
        yaw = info["initial_yaw_deg_map"]  # 지도 기준 헤딩이다.
        x -= math.sin(math.radians(yaw)) * offset_m  # 초기 횡오프셋의 X 성분이다.
        y += math.cos(math.radians(yaw)) * offset_m  # 초기 횡오프셋의 Y 성분이다.
        self.actor_call("SetTransform", "SetTransformParam", transform={"location": {"x": x, "y": y, "z": z}, "rotation": {"x": 0.0, "y": 0.0, "z": yaw}})  # 배치한다.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:  # 배치 후 노면에 정착시킨다.
            for _ in range(60):  # 약 1.2초 동안 정차를 유지한다.
                sock.sendto(encode_control(0.0, 0.0, 1.0), (config["network"]["morai_ip"], config["control"]["host_port"]))  # 제동이다.
                time.sleep(0.02)  # 제어 주기다.
        state = self.state()  # 배치 결과를 읽는다.
        location = state["transform"]["location"]  # 실제 차량 좌표다.
        if math.hypot(location["x"] - x, location["y"] - y) > 0.5:  # 잘못 배치된 상태로 시작하지 않는다.
            raise RuntimeError("MORAI start-pose readback mismatch")  # 수집을 중단한다.
        return state  # 실제 시작 상태를 기록할 수 있게 반환한다.
