"""Small MORAI gRPC adapter using the bundled official SDK descriptors.

assets/morai_api.desc was extracted from MORAI-DriveExample_GRPC's generated
MoraiStandardProtos C# descriptors (the existing local SDK), in dependency order.
This adapter only controls the simulator at localhost; no learning code.
"""
from pathlib import Path

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.json_format import MessageToDict


class Simulator:
    def __init__(self, target="127.0.0.1:7789"):
        files = descriptor_pb2.FileDescriptorSet()
        project_root = Path(__file__).resolve().parents[2]
        files.ParseFromString((project_root / "assets/morai_api.desc").read_bytes())
        self.pool = descriptor_pool.DescriptorPool()
        for file in files.file:
            self.pool.Add(file)
        self.channel = grpc.insecure_channel(target)
        self.actor = self.msg("morai_sim_api.ObjectInfo", id={"value": "Ego"},
                              object_type=1, client_key="kcity_test")

    def msg(self, name, **kw):
        return message_factory.GetMessageClass(self.pool.FindMessageTypeByName(name))(**kw)

    def call(self, service, method, request, timeout=3):
        desc = self.pool.FindServiceByName("morai_sim_api." + service).methods_by_name[method]
        cls = message_factory.GetMessageClass(desc.output_type)
        rpc = self.channel.unary_unary("/morai_sim_api.{}/{}".format(service, method),
                    request_serializer=lambda m: m.SerializeToString(), response_deserializer=cls.FromString)
        return rpc(request, timeout=timeout)

    def actor_call(self, method, message=None, **kw):
        req = self.actor if message is None else self.msg("morai_sim_api.actor." + message,
                                                         actor_info=self.actor, **kw)
        result = self.call("actor.Actor", method, req)
        if hasattr(result, "status") and result.status != 1:
            raise RuntimeError("{}: {}".format(method, result))
        return result

    def state(self):
        return MessageToDict(self.actor_call("GetActorState"), preserving_proto_field_name=True)

    def teleport(self, x, y, z, yaw):
        self.actor_call("SetVelocity", "SetVelocityParam", velocity=0.)
        return self.actor_call("SetTransform", "SetTransformParam",
            transform={"location": {"x": x, "y": y, "z": z},
                       "rotation": {"x": 0., "y": 0., "z": yaw}})

    def automatic(self):
        self.actor_call("SetVehicleControlMode", "VehicleControlModeParam", mode=3)
        self.actor_call("SetVehicleDynamicsSpeedLimit", "VehicleDynamicsSpeedLimit",
                        speed_limiter=True, speed_limit=20., reset=False)

    def resume(self):
        self.call("simulation.Simulation", "Resume", self.msg("morai_sim_api.Empty"))

    def conditions(self):
        self.call("environment.Environment", "SetWeather",
                  self.msg("morai_sim_api.environment.Weather", type=0, instantly=True))
        self.call("environment.Environment", "SetTime",
                  self.msg("morai_sim_api.environment.SimulationTime", hour=12, instantly=True))
