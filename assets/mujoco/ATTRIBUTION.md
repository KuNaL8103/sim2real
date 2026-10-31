# Third-party asset attribution: MuJoCo Menagerie Franka Panda

- Upstream repository: <https://github.com/google-deepmind/mujoco_menagerie>
- Pinned commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
  (commit date 2026-07-04, subject "Clean up scene <statistic> tags (#298)")
- Retrieved: 2026-10-31 with `scripts/fetch_assets.sh`
- Upstream path: `franka_emika_panda/`
- Upstream project: MuJoCo Menagerie, Copyright Google DeepMind and contributors.

## License actually found in the pinned folder

`franka_emika_panda/LICENSE` at the pinned commit starts with:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```

That is the Apache License, Version 2.0. The complete upstream license text is
vendored verbatim at `assets/mujoco/franka_emika_panda/LICENSE`. This project's
own `LICENSE` is the same Apache-2.0 text. No endorsement by Google DeepMind is
implied.

## What is committed and what is fetched

Committed (small text files):

- `panda.xml`, `hand.xml`, `scene.xml`, `panda_nohand.xml`, `mjx_*.xml`
- `README.md`, `CHANGELOG.md`, `LICENSE`
- `assets/mujoco/scene_tabletop.xml` (this project's scene, which includes the
  unmodified upstream `franka_emika_panda/panda.xml`)

Git-ignored and restored by `scripts/fetch_assets.sh`:

- `assets/mujoco/franka_emika_panda/assets/` (about 33 MB of OBJ/STL meshes)
- `assets/mujoco/franka_emika_panda/panda.png` (not needed)

The wrist camera is added at compile time with MuJoCo `MjSpec`, and the table,
three cubes, two target zones, and overhead camera live in
`assets/mujoco/scene_tabletop.xml`. The upstream MJCF is not edited.
