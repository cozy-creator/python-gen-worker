# Terminology

This repo uses GitHub/Hugging Face style naming for namespaces.

## Namespaces

- **owner**: the namespace part of `owner/repo`. An owner is either an organization or an individual user.

## Identifiers

- **model repo**: `owner/repo`
- **published endpoint**: `owner/project_name/endpoint_name`

## Auth Mapping

- AuthKit JWT claim: `org` (wire compatibility)
- Product terminology: `owner`

In code and logs, prefer calling the namespace `owner` when referring to `owner/repo` or `owner/project/endpoint`.

