# Static web release inputs

`web/` contains the studio and the Python package index, but it is not the
complete published site. Distribution archives and some other published assets
are separate release inputs. Publishing only `web/` can remove existing paths
or leave the package index pointing at missing downloads.

Before a release, assemble the full site in a staging directory and obtain a
fresh `files` mapping from the current Cloudflare Pages deployment and save the
current `/simple/wedeo/` HTML as a reference. Run:

```sh
python3 scripts/check_web_release.py --site /path/to/staged-site --baseline /path/to/current-deployment.json --reference-index /path/to/current-package-index.html
```

The check rejects missing baseline paths or headers, a missing Wedeo project
link in the package index root, removal or alteration of existing package links, unsafe links and incorrect checksums, names, versions or Python-version requirements for referenced artifacts. To intentionally remove or update existing links, review the changes and use `--allow-index-removal`; the check reports the removed links. Filename/version comparison is conservative; unusual valid version spellings need manual review. It
does not upload, compare every other asset byte, or prove that `_headers` and
routing match production. Review those separately before deploying.
