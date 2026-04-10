## Qellimi

Ky bundle eshte pergatitur qe te ngarkohet ne ChatGPT per nje auditim te projektit GreenNet.

ChatGPT duhet te verifikoje:

1. A perputhet implementimi me kerkesat e projektit dhe dokumentet e propozimit.
2. A ekzistojne realisht ne repo artefaktet, raportet, testet dhe komponentet qe permenden ne dokumentacion.
3. A ka mospërputhje midis pretendimeve ne dokumente dhe gjendjes aktuale te kodit.
4. Cilat kerkesa duken te plotesuara, pjeserisht te plotesuara, ose jo te plotesuara.
5. Cilat jane boshllëqet, risket, ose path-et e permendura ne dokumente qe mungojne ne bundle.

## Cfare perfshin bundle

- kodin kryesor Python
- testet
- frontend-in pa `node_modules` dhe pa `dist`
- dokumentacionin kryesor
- dokumentet e propozimit
- evidencen e auditimit final dhe bundle-t finale te evaluimit

## Prompt i sugjeruar per ChatGPT

```text
Analizo kete projekt si auditor teknik. Duke u bazuar te dokumentet e propozimit, README, docs/final_submission_overview.md, final_audit/*, kodi burimor, testet dhe evidenca e evaluimit, me jep:

1. nje tabele me te gjitha kerkesat ose objektivat kryesore te projektit
2. per secilen, statusin: Plotesuar / Pjeserisht / Jo i plotesuar
3. evidencen konkrete me file path
4. mospërputhjet midis dokumentacionit dhe repo-s aktuale
5. cfare mungon per ta quajtur projektin plotesisht ne rregull per dorezim

Behu strikt: nese nje dokument pretendon nje file ose artifact qe nuk ekziston ne bundle, shenoje si mospërputhje.
```
