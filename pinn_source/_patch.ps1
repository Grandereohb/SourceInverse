$path = "C:\Document\博士\高老师项目\溯源\pinn_source\pinn_source_pinn.py"
$text = Get-Content -Raw $path
$needle = 'print("Estimated source (lat,lon):", pred_lat, pred_lon)' + "`n"
$insert = @"
print(\"Estimated source (lat,lon):\", pred_lat, pred_lon)

    # --------- Visualization ---------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(sites[\"x\"], sites[\"y\"], c=\"blue\", s=80, label=\"Stations\")
    for _, r in sites.iterrows():
        plt.text(r[\"x\"], r[\"y\"], str(r[\"station\"]), fontsize=10, ha=\"left\", va=\"bottom\")
    plt.scatter(xs, ys, c=\"red\", s=150, marker=\"*\", label=\"Estimated Source\")
    plt.axhline(0, color=\"#dddddd\", linewidth=1)
    plt.axvline(0, color=\"#dddddd\", linewidth=1)
    plt.xlabel(\"x (m)\")
    plt.ylabel(\"y (m)\")
    plt.title(\"Stations and Estimated Source\")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"@
if ($text -notlike "*$($needle.TrimEnd())*") { throw "needle not found" }
$text = $text.Replace($needle, $insert)
Set-Content -Encoding UTF8 $path $text
