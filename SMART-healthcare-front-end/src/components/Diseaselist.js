import axios from "axios";
import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom";
import Navbar from "./Navbar";
import image from "../assets/image4.jpg";

export default function Diseaselist() {
  const [data, setData] = useState([]);
  const [search, setSerach] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const results = await axios.get(
      `http://localhost:8080/api/v1/diseases/findall`
    );
    setData(results.data);
    console.log(results);
  };

  return (
    <div
      style={{
        backgroundImage: `url(${image})`,
        height: "auto",
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <Navbar />
      <div className="container">
        <div className="py-4">
          <input
            type={"text"}
            className="txt border shadow"
            placeholder="Search here"
            onChange={(e) => {
              setSerach(e.target.value);
            }}
          ></input>
          <h2 className="text-center">List View</h2>
          <table className="table border shadow">
            <thead>
              <tr>
                <th scope="col">ID</th>
                <th scope="col">Name</th>
                <th scope="col">Type</th>
                <th scope="col">Specialist</th>
                <th scope="col">Action</th>
              </tr>
            </thead>
            <tbody>
              {data
                .filter((disease) => {
                  if (search == "") {
                    return disease;
                  } else if (
                    disease.name.toLowerCase().includes(search.toLowerCase())
                  ) {
                    return disease;
                  }
                })
                .map((disease, index) => (
                  <tr>
                    <th scope="row" key={index}>
                      {index + 1}
                    </th>
                    <td>{disease.name}</td>
                    <td>{disease.type}</td>
                    <td>{disease.specialist}</td>
                    <td>
                      <Link
                        className="btn btn-success my-2 mx-6"
                        to={`/contact/${disease.specialist}`}
                      >
                        Find your doctor
                      </Link>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
