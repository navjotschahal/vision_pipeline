"""Select articulated links from a pose without inventing missing joints."""

from __future__ import annotations

from dataclasses import dataclass

from vision_pipeline.perception.pose.contracts import HumanJoint, HumanPose2D, HumanPose3D


@dataclass(frozen=True, slots=True)
class LimbChain:
    name: str
    joints: tuple[HumanJoint, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or len(self.joints) < 2:
            raise ValueError("a limb chain needs a name and at least two joints")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError("chain joints must be unique")

    @property
    def links(self) -> tuple[tuple[HumanJoint, HumanJoint], ...]:
        return tuple(zip(self.joints[:-1], self.joints[1:], strict=True))


LEFT_ARM = LimbChain(
    "left_arm", (HumanJoint.LEFT_SHOULDER, HumanJoint.LEFT_ELBOW, HumanJoint.LEFT_WRIST)
)
RIGHT_ARM = LimbChain(
    "right_arm", (HumanJoint.RIGHT_SHOULDER, HumanJoint.RIGHT_ELBOW, HumanJoint.RIGHT_WRIST)
)


def visible_links(
    pose: HumanPose2D | HumanPose3D, chain: LimbChain, *, minimum_confidence: float = 0.6
) -> tuple[tuple[HumanJoint, HumanJoint], ...]:
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between zero and one")
    return tuple(
        (start, end)
        for start, end in chain.links
        if (first := pose.get(start)) is not None
        and (second := pose.get(end)) is not None
        and first.confidence >= minimum_confidence
        and second.confidence >= minimum_confidence
    )


__all__ = ["LimbChain", "LEFT_ARM", "RIGHT_ARM", "visible_links"]
