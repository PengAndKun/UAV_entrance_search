from __future__ import annotations

import sys

from obstacle_avoidance.collect_route_episodes import main as _main


def main() -> None:
    if "--data-root" not in sys.argv:
        sys.argv.extend(["--data-root", "obstacle_avoidance_2_data"])
    _main()


if __name__ == "__main__":
    main()
