import folium

# 中心点
m = folium.Map(location=[43.8256, 125.3245], zoom_start=12)

# 添加点
folium.Marker([43.8256, 125.3245], popup="监测点1").add_to(m)
folium.Marker([43.8231, 125.3312], popup="监测点2").add_to(m)

# 保存
m.save("map.html")
