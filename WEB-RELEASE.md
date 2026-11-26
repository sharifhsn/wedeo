# Static web release inputs

`web/` contains the studio and the Python package index, but it is not the
complete published site. Distribution archives and some other published assets
are separate release inputs. Publishing only `web/` can remove existing paths
or leave the package index pointing at missing downloads.

Before a release, assemble the full site in a staging directory and obtain a
fresh `files` mapping from the current Cloudflare Pages deployment. Run:

```sh
python3 scripts/check_web_release.py --site /path/to/staged-site --baseline /path/to/current-deployment.json
```

The check rejects missing baseline paths or headers, a missing Wedeo project
link in the package index root, unsafe links and incorrect checksums or Python-version requirements for artifacts referenced by the Python package index. It
does not upload, compare every other asset byte, or prove that `_headers` and
routing match production. Review those separately before deploying.
